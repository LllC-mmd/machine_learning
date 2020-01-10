# An encoder-decoder model with adversarial module for 1D signal denoising
import sklearn.utils
import os
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.utils
from torch.utils import data


class ToVariable(object):
    def __call__(self, sample):
        s = {}
        for k, v in sample.items():
            s[k] = torch.autograd.Variable(sample[k])
        return s


class ToDevice(object):
    def __init__(self, device):
        assert isinstance(device, int)
        self.device = device

    def __call__(self, sample):
        s = {}
        for k in sample:
            if self.device > 0:
                s[k] = sample[k].cuda(self.device)
            else:
                s[k] = sample[k].cpu()
        return s


class SigDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, file_list, vid, sample_length, total_length, transform=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transform
        self.vid = vid
        self.sl = sample_length
        self.Tl = total_length
        self.n_sample = int(self.Tl/self.sl)

    def __len__(self):
        return len(self.file_list)*self.n_sample

    def __getitem__(self, idx):
        f_idx = int(idx/self.n_sample)
        a_idx = idx - f_idx * self.n_sample
        sigdata = np.loadtxt(os.path.join(self.root_dir, self.file_list[f_idx]), dtype=np.double)[a_idx*self.sl:(a_idx+1)*self.sl, self.vid]
        # sigdata = ToTensor()(sigdata)
        sample = {"sig": sigdata}
        if self.transform:
            sample = self.transform(sample)
        return sample


class SigEncoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SigEncoder, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        #self.bn = nn.BatchNorm1d(input_channel)
        self.cov1 = nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, padding=1, dilation=1)
        self.cov2 = nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, padding=3, dilation=3)
        self.cov3 = nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, padding=3, dilation=3)
        self.cov4 = nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, padding=6, dilation=6)

    def forward(self, sig):
        # sig' shape: [batch_size, C, L]
        resblock = []
        sig_out = self.lrelu(self.cov1(sig))
        resblock.append(sig_out)
        sig_out = self.lrelu(self.cov2(sig_out))
        resblock.append(sig_out)
        sig_out = self.lrelu(self.cov3(sig_out))
        resblock.append(sig_out)
        sig_out = self.lrelu(self.cov4(sig_out))
        return sig_out, resblock


class SigDecoder(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(SigDecoder, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.decov1 = nn.ConvTranspose1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=6, dilation=6)
        self.cov1 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=1, dilation=1)
        self.decov2 = nn.ConvTranspose1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=3, dilation=3)
        self.cov2 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=1, dilation=1)
        self.decov3 = nn.ConvTranspose1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=3, dilation=3)
        self.cov3 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=1, dilation=1)
        self.cov4 = nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, padding=0, dilation=1)

    def forward(self, sig, resBlock):
        # resBlock' shape: [3, batch_size, C, L]
        sig_out = self.decov1(sig)
        sig_out = self.lrelu(sig_out+resBlock[2])
        sig_out = self.lrelu(self.cov1(sig_out))
        sig_out = self.decov2(sig_out)
        sig_out = self.lrelu(sig_out+resBlock[1])
        sig_out = self.lrelu(self.cov2(sig_out))
        sig_out = self.lrelu(sig_out+resBlock[0])
        sig_out = self.lrelu(self.cov3(sig_out))
        sig_out = self.lrelu(self.cov4(sig_out))
        return sig_out


class Discriminator(nn.Module):
    def __init__(self, input_channel, input_length, hidden_size):
        super(Discriminator, self).__init__()
        self.cov1 = nn.Conv1d(in_channels=input_channel, out_channels=1, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(input_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), 1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sig):
        # resBlock' shape: [3, batch_size, C, L]
        sig_out = self.cov1(sig)
        sig_out = sig_out.reshape(sig_out.size(0), -1)
        sig_out = self.lrelu(self.fc1(sig_out))
        sig_out = self.lrelu(self.fc2(sig_out))
        #p = self.fc2(sig_out)
        p = self.sigmoid(self.fc3(sig_out))
        return p


def train(encoder, decoder, discriminator, encoder_optimizer, decoder_optimizer, discriminator_optimizer, v):
    # Model configuration
    config = {"--batch_size": 20, "--root_dir": "TestWEATHnoise_N-5_T-2000", "--gpu": 0, "--maxiter": 400,
              "--vid": v, "--sample_length": 50, "--total_length": 2000, "--n_epoch": 20, "--cut_value": 0.01}
    batch_size = config["--batch_size"]
    root_dir = config["--root_dir"]
    gpu = config["--gpu"]
    maxiter = config["--maxiter"]
    sl = config["--sample_length"]
    tl = config["--total_length"]
    vid = config["--vid"]
    nepoch = config["--n_epoch"]
    cut_value = config["--cut_value"]
    print("Epoch\tBatch\tDiscriminatorLoss\tEncoder-DecoderLoss\tMSEloss")
    for e in range(0, nepoch):
        # set dataset
        data_df = pd.DataFrame(pd.read_excel("data/data_ref.xlsx"))
        data_df = sklearn.utils.shuffle(data_df)
        file_list = np.array(data_df["sig_list"].values)
        sigdata = SigDataSet(root_dir=root_dir, file_list=file_list, vid=vid, sample_length=sl, total_length=tl)
        dataloader = torch.utils.data.DataLoader(sigdata, batch_size=batch_size, num_workers=1)
        dataloader = iter(dataloader)
        # train iteration
        Tensor = torch.cuda.DoubleTensor if gpu else torch.DoubleTensor
        mseloss = torch.nn.MSELoss(reduction='mean')
        binClassloss = torch.nn.BCELoss(reduction='mean')  # (input, target)
        for i in range(0, maxiter):
            # print("Train time: %d" % i)
            sample = ToVariable()(ToDevice(gpu)(next(dataloader)))
            insig = torch.unsqueeze(Tensor(sample["sig"]), 1)
            # encoder-decoder pass
            sig_en, sig_res = encoder(insig)
            sig_de = decoder(sig_en, sig_res)
            # Optimization for parameters
            # Step 1: train encoder-decoder, minimize the difference between clean signal and original signal
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            ed_loss = mseloss(sig_de, insig) - binClassloss(discriminator(sig_en), Tensor(insig.size(0), 1).fill_(0.0))
            #ed_loss = mseloss(sig_de, insig) + mseloss(discriminator(sig_en), Tensor(insig.size(0), 1).fill_(1.0))
            #ed_loss = mseloss(sig_de, insig) - torch.mean(discriminator(sig_en))
            ed_loss.backward(retain_graph=True)
            encoder_optimizer.step()
            decoder_optimizer.step()
            if i % 5 == 0:
                # Step 2: train discriminator, maximize the difference between clean signal (1) and noise (0)
                discriminator_optimizer.zero_grad()
                en_sig_de, res = encoder(sig_de)
                d_loss = binClassloss(discriminator(sig_en), Tensor(insig.size(0), 1).fill_(0.0))\
                        + binClassloss(discriminator(en_sig_de), Tensor(insig.size(0), 1).fill_(1.0))
                #d_loss = mseloss(discriminator(sig_en), Tensor(insig.size(0), 1).fill_(0.0)) \
                      #+ mseloss(discriminator(en_sig_de), Tensor(insig.size(0), 1).fill_(1.0))
                #d_loss = -torch.mean(discriminator(en_sig_de))+torch.mean(discriminator(sig_en))
                d_loss.backward()
                discriminator_optimizer.step()
                #for p in discriminator.parameters():
                    #p.data.clamp_(-cut_value, cut_value)
                print(str(e)+"\t"+str(i)+"\t"+str(d_loss.item())+"\t"+str(ed_loss.item())+"\t"+str(mseloss(sig_de, insig).item()))
    torch.save(encoder.state_dict(), "para_backup/W_5_2000_encoder_"+str(vid)+".pkl")
    torch.save(decoder.state_dict(), "para_backup/W_5_2000_decoder_" + str(vid) + ".pkl")


def ed_denoise(sig, ds, v):
    encoder = SigEncoder(1, 32)
    encoder.to(torch.double)
    decoder = SigDecoder(32, 1)
    decoder.to(torch.double)
    # denoise configuration
    config = {"--vid": v, "--sample_length": 50, "--total_length": 2000, "--gpu": 0}
    sl = config["--sample_length"]
    tl = config["--total_length"]
    vid = config["--vid"]
    gpu = config["--gpu"]
    encoder.load_state_dict(torch.load("para_backup/"+str(ds)+"_encoder_"+str(vid)+".pkl"))
    decoder.load_state_dict(torch.load("para_backup/"+str(ds) + "_decoder_" + str(vid) + ".pkl"))
    n = int(tl/sl)
    sig_arr = []
    Tensor = torch.cuda.DoubleTensor if gpu else torch.DoubleTensor
    for i in range(0, n):
        insig = torch.unsqueeze(Tensor(sig[i*sl:(i+1)*sl, vid]), 0)
        insig = torch.unsqueeze(insig, 0)
        sig_en, sig_res = encoder(insig)
        sig_de = decoder(sig_en, sig_res)
        sig_arr.append(np.squeeze(sig_de.detach().cpu().numpy()))
    sig_arr = np.array(sig_arr).reshape(-1)
    return sig_arr


if __name__ == '__main__':
    for i in range(0, 10):
        encoder = SigEncoder(1, 32)
        encoder.to(torch.double)
        decoder = SigDecoder(32, 1)
        decoder.to(torch.double)
        dis = Discriminator(32, 50, 64)
        dis.to(torch.double)
        #print(encoder.parameters)
        e_opt = torch.optim.Adam(encoder.parameters(), lr=5e-5, betas=(0.9, 0.99), amsgrad=True)
        #e_opt = torch.optim.RMSprop(encoder.parameters(), lr=5e-5)
        d_opt = torch.optim.Adam(decoder.parameters(), lr=5e-5, betas=(0.9, 0.99), amsgrad=True)
        #d_opt = torch.optim.RMSprop(decoder.parameters(), lr=5e-5)
        dis_opt = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.9, 0.99), amsgrad=True)
        #dis_opt = torch.optim.RMSprop(dis.parameters(), lr=5e-5)
        train(encoder, decoder, dis, e_opt, d_opt, dis_opt, i)
    '''
    train(encoder, decoder, dis, e_opt, d_opt, dis_opt, 4)
    dir_name = "TestWEATHnoise_N-5_T-1000"
    file_name = "TestWEATHnoise_N-5_T-1000_0150.txt"
    colordict = {"0": "red", "1": "lightseagreen", "2": "navy", "3": "darkorange", "4": "crimson"}
    dta = np.loadtxt(os.path.join(dir_name, file_name))
    fig, ax = plt.subplots(5, 2, sharex="col", sharey="row", figsize=(20, 8))
    for i in range(0, 5):
        sig_arr = ed_denoise(dta, "W_5_1000", encoder, decoder, i)
        # plot
        t = np.linspace(1, len(dta[:, i]), len(dta[:, i]))
        ax[i][0].plot(t, dta[:, i], color=colordict[str(i)], linewidth=1)
        ax[i][1].plot(t, sig_arr, color=colordict[str(i)], linewidth=1)
    ax[0][0].set_title("original data")
    ax[0][1].set_title("data denoised with encoder-decoder")
    ax[0][0].set_ylabel("V1")
    ax[1][0].set_ylabel("V2")
    ax[2][0].set_ylabel("V3")
    ax[3][0].set_ylabel("V4")
    ax[4][0].set_ylabel("V5")
    ax[4][0].set_xlabel("t")
    ax[4][1].set_xlabel("t")
    #plt.show(dpi=600)
    plt.savefig("Fig_encoder-decoder.png", dpi=600)
    #plt.show()
    '''
