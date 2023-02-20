import os
import argparse
from math import log10
import h5py
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from decimal import *


def h5_read_data(filename, dataset):
    with h5py.File(filename, 'r') as f:
        dset = f[dataset]
        return dset[:]


def get_allfile(path):  # 获取所有文件
    all_file = []
    for f in os.listdir(path):  # listdir返回文件中所有目录
        f_name = os.path.join(path, f)
        all_file.append(f_name)
    return all_file


def floatrange(start, stop, steps):
    resultList = []

    while Decimal(str(start)) <= Decimal(str(stop)):
        resultList.append(float(Decimal(str(start))))

        start = Decimal(str(start)) + Decimal(str(steps))

    return resultList

def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x), -1)
    x_imag = x[:, 1, :, :].view(len(x), -1)
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1)
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1)
    power = torch.sum(x_real ** 2 + x_imag ** 2, axis=1)
    mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
    nmse = mse / power
    return nmse
class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse


def get_dataset(path_test, model_path, device, batch_size, slot=2, T=5):  # 获取所有文件

    with torch.no_grad():
        net_g = torch.load(model_path, map_location='cuda:0').to(device)
        print(net_g)

    y_test_nmse = []

    all_file_test = get_allfile(path_test)
    all_file_test.sort()
    print(all_file_test)

    for i_f in range(len(all_file_test)):
        print(i_f)
        test_data = h5_read_data(all_file_test[i_f], 'ls')
        test_label = h5_read_data(all_file_test[i_f], 'gt')

        test_data = torch.from_numpy(test_data[:, :, T - slot:T:1, :, :])
        test_label = torch.from_numpy(test_label)

        test_dataset = TensorDataset(test_data, test_label)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        criterionNMSE_test = NMSELoss(reduction='sum').to(device)  # nn.MSELoss()

        # val
        total_nmse = 0
        net_g.eval()
        with torch.no_grad():
            for batch in test_loader:
                input, target = batch[0].to(device), batch[1].to(device)
                prediction = net_g(input)
                total_nmse += criterionNMSE_test(target, prediction).item()
        average_nmse = total_nmse / len(test_dataset)
        avg_loss = 10 * log10(average_nmse)
        print("===> val_loader Avg. loss: {:.4f} dB".format(avg_loss))
        y_test_nmse.append(avg_loss)

    return y_test_nmse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='channel_estimation_test')
    parser.add_argument('--cuda', action='store_false', help='use cuda')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    path = './SISOCDL_TestData/test_SinglePilot/'
    path_test = path + 'test_30km_B_300ns_slot10'

    slot = 5  #network T
    model_path = "./checkpoint/netG_SinglePilot_T5.pth"

    T: int = 10  #dataset T
    batch_size = 64

    y_test_nmse = get_dataset(path_test, model_path, device, batch_size, slot=slot, T=T)

    x = floatrange(0, 20, 2.5)
    l1, = plt.plot(x, y_test_nmse, marker='.', color='#1661ab', linestyle='dotted')
    plt.legend(handles=[l1],
               labels=['MSTGAN'], loc='best')
    plt.xlabel("SNR(dB)")
    plt.ylabel("NMSE(dB)")
    plt.grid()
    plt.show()

    print(y_test_nmse)
    filename = open('MSTGAN_test_nmse.txt', 'a')
    for value in y_test_nmse:
        filename.write(str(value) + ',')
    filename.write('\n')
    filename.close()
