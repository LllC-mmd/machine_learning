import os,sys,contextlib
import numpy as np
import torch
import time
import yaml
import getopt
from learn_water2D_swmm import Water2DSwmmLearner


def _setoptions(options):
    assert options['--precision'] in ['float','double']
    # str options
    strtype = ['taskdescriptor', 'constraint', 'recordfile']
    options = _options_cast(options, strtype, str)
    assert options['--constraint'] in ['frozen','moment','free']
    # int options
    inttype = ['gpu', 'kernel_size', 'max_order', 'xn', 'yn', 'interp_degree', 'interp_mesh_size', 'nonlinear_interp_degree', 'nonlinear_interp_mesh_size',
            'initfreq', 'batch_size', 'teststepnum', 'maxiter', 'recordcycle', 'savecycle', 'repeatnum']
    options = _options_cast(options, inttype, int)
    # float options
    floattype = ['dt', 'start_noise_level', 'end_noise_level', 'nonlinear_interp_mesh_bound', 'diffusivity']
    options = _options_cast(options, floattype, float)

    options['--layer'] = list(_option_analytic(options['--layer'], int))
    return options

def setoptions(*, argv=None, kw=None, configfile=None, isload=False):
    """
    proirity: argv>kw>configfile
    Arguments:
        argv (list): command line options
        kw (dict): options
        configfile (str): configfile path
        isload (bool): load or set new options
    """
    options = {
            '--precision':'double',
            '--taskdescriptor':'nonlinpde0',
            '--constraint':'moment',
            '--gpu':-1,
            '--kernel_size':7,'--max_order':2,
            '--xn':'50','--yn':'50',
            '--interp_degree':2,'--interp_mesh_size':5,
            '--nonlinear_interp_degree':2, '--nonlinear_interp_mesh_size':20,
            '--nonlinear_interp_mesh_bound':15,
            '--initfreq':4,'--diffusivity':0.3,'--nonlinear_coefficient':15,
            '--batch_size':24,'--teststepnum':80,
            '--maxiter':20000,
            '--dt':1e-2,
            '--start_noise_level':0.01,'--end_noise_level':0.01,
            '--layer':list(range(0,21)),
            '--recordfile':'convergence',
            '--recordcycle':200,'--savecycle':10000,
            '--repeatnum':25,
            }
    longopts = list(k[2:]+'=' for k in options)
    longopts.append('configfile=')
    if not argv is None:
        options.update(dict(getopt.getopt(argv, shortopts='f',longopts=longopts)[0]))
    if '--configfile' in options:
        assert configfile is None, 'duplicate configfile in argv.'
        configfile = options['--configfile']
    if not configfile is None:
        options['--configfile'] = configfile
        with open(configfile, 'r') as f:
            options.update(yaml.load(f))
    if not kw is None:
        options.update(kw)
    if not argv is None:
        options.update(dict(getopt.getopt(argv, shortopts='f',longopts=longopts)[0]))
    options = _setoptions(options)
    options.pop('-f',1)
    savepath = 'checkpoint/'+options['--taskdescriptor']
    if not isload:
        try:
            os.makedirs(savepath)
        except FileExistsError:
            os.rename(savepath, savepath+'-'+str(np.random.randint(2**32)))
            os.makedirs(savepath)
        with open(savepath+'/options.yaml', 'w') as f:
            print(yaml.dump(options), file=f)
    return options


class callbackgen(object):
    def __init__(self, options, nfi=None, module=None, stage=None):
        self.taskdescriptor = options['--taskdescriptor']
        self.recordfile = options['--recordfile']
        self.recordcycle = options['--recordcycle']
        self.savecycle = options['--savecycle']
        self.savepath = 'checkpoint/'+self.taskdescriptor
        self.startt = time.time()
        self.Fs = []
        self.Gs = []
        self.ITERNUM = 0

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, v):
        self._stage = v
        with self.open() as output:
            print('\n', file=output)
            print('current stage is: '+v, file=output)

    @contextlib.contextmanager
    def open(self):
        isfile = (not self.recordfile is None)
        if isfile:
            output = open(self.savepath+'/'+self.recordfile, 'a')
        else:
            output = sys.stdout
        try:
            yield output
        finally:
            if isfile:
                output.close()
    
    # remember to set self.nfi,self.module,self.stage
    def save(self, xopt, iternum):
        self.nfi.flat_params = xopt
        try:
            os.mkdir(self.savepath+'/params')
        except:
            pass
        filename = self.savepath+'/params/xopt-'+str(iternum)
        torch.save(self.module.state_dict(), filename)
        return None

    def load(self, file_name):
        filename = self.savepath+'/params/'+file_name
        params = torch.load(filename)
        self.module.load_state_dict(params)
        return None

    def record(self, xopt, iternum, **args):
        self.Fs.append(self.nfi.f(xopt))
        self.Gs.append(np.linalg.norm(self.nfi.fprime(xopt)))
        stopt = time.time()
        with self.open() as output:
            print('iter:{:6d}'.format(iternum), '   time: {:.2f}'.format(stopt-self.startt), file=output)
            print('Func: {:.2e}'.format(self.Fs[-1]), ' |Func_prime|: {:.2e}'.format(self.Gs[-1]), file=output)
        self.startt = stopt
        return None

    def __call__(self, xopt, **args):
        if self.ITERNUM % self.recordcycle == 0:
            self.record(xopt, iternum=self.ITERNUM, **args)  # record function and differential value
        if self.ITERNUM % self.savecycle == 0:
            self.save(xopt, iternum=self.ITERNUM)  # save checkpoints
        self.ITERNUM += 1
        return None


def setenv(options):
    namestobeupdate = {}
    namestobeupdate['kernel_size'] = [options['--kernel_size'], ] * 2
    namestobeupdate['h_max_order'] = options['--h_max_order']
    namestobeupdate['u_max_order'] = options['--u_max_order']
    namestobeupdate['dx'] = options['--dx']
    namestobeupdate['dt'] = options['--dt']
    namestobeupdate['constraint'] = options['--constraint']
    namestobeupdate['gpu'] = options['--gpu']
    namestobeupdate['precision'] = options['--precision']

    PdeLearner = Water2DSwmmLearner(kernel_size=namestobeupdate['kernel_size'], h_max_order=namestobeupdate['h_max_order'],
                                    u_max_order=namestobeupdate['u_max_order'],dx=namestobeupdate['dx'], dt=namestobeupdate['dt'],
                                    constraint=namestobeupdate['constraint'])
    if namestobeupdate['precision'] == 'double':
        PdeLearner.double()
    else:
        PdeLearner.float()

    if namestobeupdate['gpu'] >= 0:
        PdeLearner.cuda(namestobeupdate['gpu'])
    else:
        PdeLearner.cpu()
    callback = callbackgen(options) # some useful interface

    return namestobeupdate, callback, PdeLearner
