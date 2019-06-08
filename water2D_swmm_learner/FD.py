"""Finite Difference"""
"""Ref to
@inproceedings{long2018pde,
    title={PDE-Net: Learning PDEs from Data},
    author={Long, Zichao and Lu, Yiping and Ma, Xianzhong and Dong, Bin},
    booktitle={Proceedings of the 35th International Conference on Machine Learning (ICML 2018)},
    year={2018}
}
"""
import numpy as np
from functools import reduce
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import MK

__all__ = ['MomentBank','FD2d']


def _inv_equal_order_m(d,m):
    A = []
    assert d >= 1 and m >= 0
    if d == 1:
        A = [[m,],]
        return A
    if m == 0:
        for i in range(d):
            A.append(0)
        return [A,]
    for k in range(m+1):
        B = _inv_equal_order_m(d-1,m-k)
        for b in B:
            b.append(k)  # change B's element
        A = A+B
    return A


def _less_order_m(d,m):
    A = []
    for k in range(m+1):
        B = _inv_equal_order_m(d,k)
        for b in B:
            b.reverse()
        B.sort()
        B.reverse()
        A.append(B)
    return A


# _torch_setter_by_index(self.moment.data[i], o, 1)
def _torch_setter_by_index(t,i,v):
    for j in i[:-1]:
        # 抽丝剥茧式操作
        t = t[j]  # t is a local variable
    t[i[-1]] = v


def _torch_reader_by_index(t,i):
    for j in i:
        t = t[j]
    return t


class MomentBank(nn.Module):
    """
    generate moment matrix bank for differential kernels with order 
    no more than max_order.
    Arguments:
        dim (int): dimension
        kernel_size (tuple of int): size of differential kernels
        max_order (int): max order of differential kernels
        dx (double): the MomentBank.kernel will automatically compute kernels 
            according to MomentBank.moment and MomentBank.dx
        constraint (string): 'moment' or 'free'. Determine MomentBank.x_proj 
            and MomentBank.grad_proj
    """
    def __init__(self, dim, kernel_size, max_order, dx, constraint='moment'):
        super(MomentBank, self).__init__()
        self._dim = dim
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,]*dim
        assert min(kernel_size) > max_order
        self.m2k = MK.M2K(kernel_size)
        self._kernel_size = kernel_size.copy()
        self._max_order = max_order
        if not np.iterable(dx):
            dx = [dx,]*dim
        self._dx = dx.copy()
        self.constraint = constraint
        d = dim
        m = max_order
        self._order_bank = _less_order_m(d, m)
        N = 0
        for a in self._order_bank:
            N += len(a)
        self.N = N
        moment = torch.zeros(*([N,]+kernel_size))  # torch.size: (N, *kernel_size)
        index = np.zeros([m+1,]*dim,dtype=np.int64)
        for i,o in enumerate(self.flat_order_bank()):
            _torch_setter_by_index(moment[i],o,1)  # we use N kernels to approximate N differential operators
            _torch_setter_by_index(index,o,i)
            # moment[i,*o] = 1
        self.moment = nn.Parameter(moment)
        self._index = index
        scale = torch.ones(self.moment.size()[0])
        l = lambda a,b:a*b
        for i,o in enumerate(self.flat_order_bank()):
            s = reduce(l, (self.dx[j]**oj for j,oj in enumerate(o)), 1)
            scale[i] = 1/s
        self.register_buffer('scale',scale)


    def __index__(self,*args):
        return self.moment[_torch_reader_by_index(self._index, args)]

    def dim(self):
        return self._dim

    @property
    def dx(self):
        return self._dx.copy()

    def kernel(self):
        scale = Variable(self.scale[:,np.newaxis])
        kernel = self.m2k(self.moment)  # torch.size: (N, *kernel_size)
        size = kernel.size()
        kernel = kernel.view([size[0],-1])
        return (kernel*scale).view(size)[:,np.newaxis]

    def flat_order_bank(self):
        for a in self._order_bank:
            for o in a:
                yield o  # o is a vector which has d elements, d is dim, sum of o <= acc, e.g. [0, 0]

    def _proj_(self, M, s, c):
        for j in range(s):
            for o in self._order_bank[j]:
                _torch_setter_by_index(M,o,c)
                # M[*o] = c, * is unzip operation

    def x_proj(self,*args,**kw):
        if self.constraint == 'free':
            return None
        if isinstance(self.constraint,int):
            acc = self.constraint
        else:
            acc = 1
        for i,o in enumerate(self.flat_order_bank()):
            self._proj_(self.moment.data[i],sum(o)+acc,0)
            # self.moment.data[i, j, k] = 0, where j+k <= sum(o)
            _torch_setter_by_index(self.moment.data[i], o, 1)
            # self.moment.data[i,*o] = 1, * is unzip operation
        return None

    def grad_proj(self,*args,**kw):
        if self.constraint == 'free':
            return None
        if isinstance(self.constraint,int):
            acc = self.constraint
        else:
            acc = 1
        for i,o in enumerate(self.flat_order_bank()):
            self._proj_(self.moment.grad.data[i],sum(o)+acc,0)
        return None

    def forward(self):
        return self.kernel()


class _FDNd(nn.Module):
    """
    Finite difference automatically handle boundary conditions
    Arguments for class:`_FDNd`:
        dim (int): dimension
        kernel_size (tuple of int): finite difference kernel size
        boundary (string): 'Dirichlet' or 'Periodic'
    Arguments for class:`MomentBank`:
        max_order, dx, constraint
    """
    def __init__(self, dim, kernel_size):
        super(_FDNd, self).__init__()
        self._dim = dim
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,]*dim
        self._kernel_size = kernel_size.copy()
        padwidth = []
        for k in reversed(kernel_size):
            padwidth.append((k-1)//2)
            padwidth.append(k-1-(k-1)//2)
        self._padwidth = padwidth

    def dim(self):
        return self._dim

    @property
    def padwidth(self):
        return self._padwidth.copy()

    def conv(self, inputs, weight):
        raise NotImplementedError

    def forward(self, inputs, kernel):   # inputs: (batch_size, 1, H, W), kernel: (1, 1, 7, 7)
        # inputs = inputs[:,np.newaxis]
        return self.conv(inputs, kernel, padding=2)


class FD2d(_FDNd):
    def __init__(self, kernel_size, max_order, dx, constraint='moment'):
        super(FD2d, self).__init__(2, kernel_size)
        self.MomentBank = MomentBank(2, kernel_size, max_order, dx=dx, constraint=constraint)
        self.N = self.MomentBank.N
        self.conv = F.conv2d

