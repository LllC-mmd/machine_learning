import FD
import torch
import torch.nn.functional as F


class Water2DSwmmLearner(torch.nn.Module):
    def __init__(self, kernel_size, h_max_order, u_max_order, dx, dt, constraint):
        super(Water2DSwmmLearner, self).__init__()
        self.h_id = FD.FD2d(kernel_size=kernel_size, max_order=0, dx=dx, constraint=constraint)
        self.h_fd2d = FD.FD2d(kernel_size=kernel_size, max_order=h_max_order, dx=dx, constraint=constraint)
        self.u_id = FD.FD2d(kernel_size=kernel_size, max_order=0, dx=dx, constraint=constraint)
        self.u_fd2d = FD.FD2d(kernel_size=kernel_size, max_order=u_max_order, dx=dx, constraint=constraint)
        self.v_id = FD.FD2d(kernel_size=kernel_size, max_order=0, dx=dx, constraint=constraint)
        self.v_fd2d = FD.FD2d(kernel_size=kernel_size, max_order=u_max_order, dx=dx, constraint=constraint)
        # parameters for init normalization
        self.h_gamma = torch.nn.Parameter(torch.randn(1))
        self.h_beta = torch.nn.Parameter(torch.randn(1))
        self.u_gamma = torch.nn.Parameter(torch.randn(1))
        self.u_beta = torch.nn.Parameter(torch.randn(1))
        self.v_gamma = torch.nn.Parameter(torch.randn(1))
        self.v_beta = torch.nn.Parameter(torch.randn(1))
        # dt
        self.register_buffer("dt", torch.FloatTensor(1).fill_(dt))
        in_channels = self.h_id.N + self.h_fd2d.N + self.u_id.N + self.u_fd2d.N + self.v_id.N + self.v_fd2d.N
        # when max_order = 2, self.h_id.N = 1, self.h_fd2d.N = 6
        # when max_order = 3, self.h_id.N = 1, self.h_fd2d.N = 15
        # h kernel
        self.h_kernel11 = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        torch.nn.init.xavier_uniform(self.h_kernel11.weight)
        self.h_kernel33 = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform(self.h_kernel33.weight)
        # u kernel
        self.u_kernel11 = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        torch.nn.init.xavier_uniform(self.u_kernel11.weight)
        self.u_kernel33 = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform(self.u_kernel33.weight)
        # v kernel
        self.v_kernel11 = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        torch.nn.init.xavier_uniform(self.v_kernel11.weight)
        self.v_kernel33 = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform(self.v_kernel33.weight)

    def params(self):
        params_list = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                # print(name)
                # print(param.size())
                params_list.append(param)
        return params_list

    def forward(self, rain, h_init, u_init, v_init, total_step):
        h_idkernel = self.h_id.MomentBank.kernel()   # torch.size(): (1, *kernel_size)
        h_fdkernel = self.h_fd2d.MomentBank.kernel()  # torch.size(): (N, *kernel_size)
        u_idkernel = self.u_id.MomentBank.kernel()
        u_fdkernel = self.u_fd2d.MomentBank.kernel()
        v_idkernel = self.v_id.MomentBank.kernel()
        v_fdkernel = self.v_fd2d.MomentBank.kernel()
        dt = torch.autograd.Variable(self.dt)
        h = h_init   # torch.size: (batch_size, 1, H, W), 1 = num_channel, H = 581, W = 860
        u = u_init
        v = v_init
        p = rain
        for i in range(0, total_step):
            hi_mean = torch.mean(h)
            hi_var = torch.mean((h-hi_mean)**2)
            ui_mean = torch.mean(u)
            ui_var = torch.mean((u-ui_mean)**2)
            vi_mean = torch.mean(v)
            vi_var = torch.mean((v-vi_mean)**2)
            # batch normalization for h, u, v
            h = self.h_gamma * (h - hi_mean) / torch.sqrt(hi_var + 1e-5) + self.h_beta
            u = self.u_gamma * (u - ui_mean) / torch.sqrt(ui_var + 1e-5) + self.u_beta
            v = self.v_gamma * (v - vi_mean) / torch.sqrt(vi_var + 1e-5) + self.v_beta
            hid = self.h_id(h, h_idkernel)   # torch.size: (batch_size, 1, H, W)
            hfd = self.h_fd2d(h, h_fdkernel)  # torch.size: (batch_size, N, H, W)
            uid = self.u_id(u, u_idkernel)
            ufd = self.u_fd2d(u, u_fdkernel)
            vid = self.v_id(v, v_idkernel)
            vfd = self.v_fd2d(v, v_fdkernel)
            # concatenate for 1*1 and 3*3 convolution
            data_union = torch.cat([hid, hfd, uid, ufd, vid, vfd], dim=1)  # torch.size: (batch_size, C, H, W)
            h = hid.squeeze(1) + dt*self.h_kernel11(data_union).sum(dim=1) + dt*self.h_kernel33(data_union).sum(dim=1) + p[:,i,:,:]*dt
            h = torch.nn.functional.relu(h-5) + 5
            h = h.unsqueeze(1)
            # u
            u = uid.squeeze(1) + dt*self.u_kernel11(data_union).sum(dim=1) + dt*self.u_kernel33(data_union).sum(dim=1) + p[:,i,:,:]*dt
            u = u.unsqueeze(1)
            # v
            v = vid.squeeze(1) + dt*self.v_kernel11(data_union).sum(dim=1) + dt*self.v_kernel33(data_union).sum(dim=1) + p[:,i,:,:]*dt
            v = v.unsqueeze(1)
        # return h.reshape(h_init.size())
        return h.reshape(h_init.size()), u.reshape(u_init.size()), v.reshape(v_init.size())
