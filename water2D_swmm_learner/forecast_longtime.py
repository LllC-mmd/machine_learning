'''
For Linux server:
import matplotlib as m
m.use('Agg')
'''
import numpy as np
import torch
import dataset
import config
import time
import matplotlib.pyplot as plt
import pandas as pd


def test_valid(root_dir, rain_list, h_list, u_list, v_list, h_sol_list, u_sol_list, v_sol_list):
    options = {'--kernel_size': 5, '--h_max_order': 2, '--u_max_order': 2, '--dx': 5, '--dt': 60,
               '--constraint': 'moment', '--gpu': 0, '--precision': 'float',
               '--taskdescriptor': 'water2D-swmm', '--recordfile': 'convergence',
               '--batch_size': 1, '--total_step': 5,
               '--test_n': 24, '--forecast_n': 80}

    namestobeupdate, callback, pde_learner = config.setenv(options)
    # load optimized parameters from .pkl file
    pde_learner.load_state_dict(torch.load("checkpoint/water2D-swmm/pdelearner_train_huv_W_lr=1e-2_bs=3.pkl"))
    callback.module = pde_learner
    test_n = options['--test_n']
    batch_size = options['--batch_size']
    forecast_n = options['--forecast_n']
    total_step = options['--total_step']
    gpu = options['--gpu']

    pdedata = dataset.PdeSolDataSet(root_dir=root_dir, rain_list=rain_list, h_list=h_list, u_list=u_list,
                                    v_list=v_list, h_sol_list=h_sol_list, u_sol_list=u_sol_list, v_sol_list=v_sol_list)
    dataloader = torch.utils.data.DataLoader(pdedata, batch_size=batch_size, num_workers=1)
    dataloader = iter(dataloader)

    start_time = time.time()
    loss = torch.nn.MSELoss()
    sample = dataset.ToVariable()(dataset.ToDevice(gpu)(next(dataloader)))
    rain = sample['rain']
    h0 = sample['h0']
    u0 = sample['u0']
    v0 = sample['v0']
    ht = sample['ht']
    for j in range(0, test_n):
        ht_learned, ut_learned, vt_learned = pde_learner(rain=rain, h_init=h0, u_init=u0, v_init=v0, total_step=total_step)
        # compute loss
        h_loss_i = loss(ht_learned, ht)
        # plot h loss img
        l = (ht_learned - ht).detach().cpu().numpy()
        l = np.squeeze(l)
        x_max = l.shape[1]
        y_max = l.shape[0]
        loss_img = l
        fig, ax = plt.subplots(1)
        plt.xlim(0, x_max)
        plt.ylim(0, y_max)
        ax.set_aspect(1)
        mesh = ax.pcolormesh(loss_img, cmap=plt.get_cmap('bwr'))
        # Ref to: https://matplotlib.org/gallery/color/colormap_reference.html
        mesh.set_clim(-10, 10)
        fig.colorbar(mesh, fraction=0.033, pad=0.04)
        fig.suptitle("Validation: $h_{%d, ws}-h_{%d, nn}$"%((j+1)*5, (j+1)*5), y=0.87)
        fig.savefig("valid_1811/loss_"+str(j)+".png", dpi=400)
        plt.close()
        # save learned ht img to npy
        np.save('valid_1811/Test_'+str(j),np.squeeze(ht_learned.detach().cpu().numpy()))
        # record time for one step simulation
        end_time = time.time()
        time_elapsed = end_time - start_time
        h_RMSE = np.mean(torch.sqrt(h_loss_i).detach().cpu().numpy())
        print('Test: %d\t\th_RMSE: %.5f\t\tTimeElapsed: %.5f' % (j, h_RMSE, time_elapsed))
        # reset time logger
        start_time = end_time
        h0 = ht_learned
        u0 = ut_learned
        v0 = vt_learned
        sample = dataset.ToVariable()(dataset.ToDevice(gpu)(next(dataloader)))
        rain = sample['rain']
        ht = sample['ht']


if __name__ == '__main__':
    data_df = pd.DataFrame(pd.read_excel("data_1811/valid_ref.xlsx"))
    rain_list = np.array(data_df["rain_list"].values)
    h_list = np.array(data_df["h_list"].values)
    u_list = np.array(data_df["u_list"].values)
    v_list = np.array(data_df["v_list"].values)
    h_sol_list = np.array(data_df["h_sol_list"].values)
    u_sol_list = np.array(data_df["u_sol_list"].values)
    v_sol_list = np.array(data_df["v_sol_list"].values)
    test_valid(root_dir="data", rain_list=rain_list, h_list=h_list, u_list=u_list, v_list=v_list, h_sol_list=h_sol_list,
               u_sol_list=u_sol_list, v_sol_list=v_sol_list)
