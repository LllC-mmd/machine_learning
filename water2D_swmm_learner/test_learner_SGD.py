# train water2D-SWMM learner using optimizing algorithms in the family of SGD
# e.g. AMSgrad, Adam
import dataset
from NFI import NumpyFunctionInterface
import torch
import torch.nn
import config
import pandas as pd
import numpy as np
import sklearn.utils


class WNRMSELoss(torch.nn.Module):
    def __init__(self):
        super(WNRMSELoss, self).__init__()

    def forward(self, o1, t1, o2, t2, o3, t3):
        size = o1.squeeze(1).size()
        loss = torch.Tensor(size[0])
        for i in range(0, size[0]):
            loss[i] = 0.5*torch.sqrt(torch.mean((o1[i].squeeze(1)-t1[i].squeeze(1))**2))/(torch.max(t1[i].squeeze(1))-torch.min(t1[i].squeeze(1)))\
                    +0.25*torch.sqrt(torch.mean((o2[i].squeeze(1)-t2[i].squeeze(1))**2))/(torch.max(t2[i].squeeze(1))-torch.min(t2[i].squeeze(1)))\
                    +0.25*torch.sqrt(torch.mean((o3[i].squeeze(1)-t3[i].squeeze(1))**2))/(torch.max(t3[i].squeeze(1))-torch.min(t3[i].squeeze(1)))
        loss = torch.mean(loss)
        return loss


def test(root_dir, rain_list, h_list, u_list, v_list, h_sol_list, u_sol_list, v_sol_list):
    # root_dir: data
    # rain_list: rain/rain_i.npy
    # h_list: h/h_i.tiff
    options = {'--kernel_size': 5, '--h_max_order': 2, '--u_max_order': 2, '--dx': 5, '--dt': 60,
               '--constraint': 'moment', '--gpu': 0, '--precision': 'float',
               '--taskdescriptor': 'water2D-swmm', '--recordfile': 'convergence',
               '--batch_size': 1, '--total_step': 5}

    namestobeupdate, callback, pde_learner = config.setenv(options)
    callback.module = pde_learner   # set callback.module
    batch_size = options['--batch_size']
    total_step = options['--total_step']
    gpu = options['--gpu']

    # set dataset
    pdedata = dataset.PdeSolDataSet(root_dir=root_dir, rain_list=rain_list, h_list=h_list, u_list=u_list,
                                    v_list=v_list, h_sol_list=h_sol_list, u_sol_list=u_sol_list, v_sol_list=v_sol_list)
    dataloader = torch.utils.data.DataLoader(pdedata, batch_size=batch_size, num_workers=1)
    dataloader = iter(dataloader)
    # Optimizer
    opt_Adam = torch.optim.Adam(pde_learner.parameters(), lr=1e-3, betas=(0.9, 0.99), amsgrad=True)
    for i in range(0, 600):
        sample = dataset.ToVariable()(dataset.ToDevice(gpu)(next(dataloader)))
        rain = sample['rain']
        h0 = sample['h0']
        u0 = sample['u0']
        v0 = sample['v0']
        ht = sample['ht']
        ut = sample['ut']
        vt = sample['vt']
        # ht_learned = pde_learner(rain=rain, h_init=h0, u_init=u0, v_init=v0, total_step=total_step)
        ht_learned, ut_learned, vt_learned = pde_learner(rain=rain, h_init=h0, u_init=u0, v_init=v0, total_step=total_step)
        # define loss function
        loss = WNRMSELoss()
        # set NumpyFunctionInterface
        def x_proj(*args,**kw):
            pde_learner.h_id.MomentBank.x_proj()
            pde_learner.h_fd2d.MomentBank.x_proj()
            pde_learner.u_id.MomentBank.x_proj()
            pde_learner.u_fd2d.MomentBank.x_proj()
            pde_learner.v_id.MomentBank.x_proj()
            pde_learner.v_fd2d.MomentBank.x_proj()
        def grad_proj(*args,**kw):
            pde_learner.h_id.MomentBank.grad_proj()
            pde_learner.h_fd2d.MomentBank.grad_proj()
            pde_learner.u_id.MomentBank.grad_proj()
            pde_learner.u_fd2d.MomentBank.grad_proj()
            pde_learner.v_id.MomentBank.grad_proj()
            pde_learner.v_fd2d.MomentBank.grad_proj()
        # optimize
        # print("Opt starting")
        nfi = NumpyFunctionInterface([dict(params=pde_learner.params(), isfrozen=False, x_proj=x_proj, grad_proj=grad_proj)],
            forward=loss, always_refresh=False)
        try:
            # mse_loss = loss(ht_learned, ht)
            nrmse_loss = loss(ht_learned, ht, ut_learned, ut, vt_learned, vt)
            mse = torch.nn.MSELoss()
            h_rmse = torch.sqrt(mse(ht_learned, ht)).detach()
            u_rmse = torch.sqrt(mse(ut_learned, ut)).detach()
            v_rmse = torch.sqrt(mse(vt_learned, vt)).detach()
            # print("Round: %d\t\tRMSE is: %e" % (i+1, torch.sqrt(mse_loss)))
            print("Round: %d\t\tNRMSE is: %e\t\th_RMSE is: %e\t\tu_RMSE is: %e\t\tv_RMSE is: %e" % (i + 1, nrmse_loss, h_rmse, u_rmse, v_rmse))
            opt_Adam.zero_grad()
            # mse_loss.backward()
            nrmse_loss.backward()
            # projected the gradient of Moment matrix to satisfy the linear constraints
            nfi.all_grad_proj()
            opt_Adam.step()
        except RuntimeError as Argument:
            with callback.open() as output:
                print(Argument, file=output)  # if overflow then just print and continue
        finally:
            # save parameters
            pass
    print(pde_learner.state_dict())
    torch.save(pde_learner.state_dict(), "checkpoint/water2D-swmm/pdelearner_train_huv_W.pkl")


if __name__ == '__main__':
    data_df = pd.DataFrame(pd.read_excel("data/data_ref.xlsx"))
    data_df = sklearn.utils.shuffle(data_df)
    rain_list = np.array(data_df["rain_list"].values)
    h_list = np.array(data_df["h_list"].values)
    u_list = np.array(data_df["u_list"].values)
    v_list = np.array(data_df["v_list"].values)
    h_sol_list = np.array(data_df["h_sol_list"].values)
    u_sol_list = np.array(data_df["u_sol_list"].values)
    v_sol_list = np.array(data_df["v_sol_list"].values)
    test(root_dir="data", rain_list=rain_list, h_list=h_list, u_list=u_list, v_list=v_list, h_sol_list=h_sol_list, u_sol_list=u_sol_list, v_sol_list=v_sol_list)
