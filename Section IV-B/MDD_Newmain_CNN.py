import copy

import scipy.io as sio                     # import scipy.io for .mat file I/O
import numpy as np                         # import numpy
#import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GraphConv,HEATConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, Dropout, ReLU, LeakyReLU, Sigmoid, Softplus,Tanh, BatchNorm1d as BN
#import MDD_net_generator as ng
from MDD_hetero_conv import MDDHeteroConv
from MDDConv import MDDNet
from MDDConv_slim import MDDSlimNet
from MDD_GarphConv import MDDGraphConv
import scipy.io as scio
import random
from torch.utils.data import TensorDataset
import os
#import h5py

class init_parameters():
    def __init__(self):
        # wireless network settings
        self.Number_of_ant = 8
        self.l_AP = 24
        self.d_MS = 6
        self.DLsub = 4
        self.ULsub = 2
        self.Sumsub = self.DLsub +self.ULsub
        self.field_length = 400
        self.shadowing_factor = 4
        self.bandwidth = 100e6
        self.carrier_f = 5e9
        self.AP_power_milli_decibel = 40
        self.AP_power = np.power(10, (self.AP_power_milli_decibel - 30) / 10)
        self.MS_power_milli_decibel = 30
        self.MS_power = np.power(10, (self.MS_power_milli_decibel - 30) / 10)
        self.noise_density_milli_decibel = -174
        self.noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.AP_SIC = 1e-12
        self.MS_SIC = 1e-11
        self.IAI_SIC = 10**-7.2
        self.IMI_SIC = 10**-4.2
        self.DL_Con = 0.5
        self.UL_Con = 0.1
        self.setting_str = "{}_APs and {}_MSs {}X{} Cell".format(self.l_AP, self.d_MS, self.field_length, self.field_length)

# def cachenew(fname, x_train):
#     h5 = h5py.File(fname, 'w')
#     h5.create_dataset('all_data', data=x_train)


def normalize_Dis_data(train_data, test_data):
    norm_train = (train_data-np.min(train_data.flatten())) / (np.max(train_data.flatten())-np.min(train_data.flatten()))
    norm_test = (test_data - np.min(test_data.flatten())) / (
                np.max(test_data.flatten()) - np.min(test_data.flatten()))
    return norm_train, norm_test
def normalize_data(train_data, test_data):
    # norm_train = (train_data-np.min(train_data.flatten())) / (np.max(train_data.flatten())-np.min(train_data.flatten()))
    # norm_test = (test_data - np.min(test_data.flatten())) / (
    #             np.max(test_data.flatten()) - np.min(test_data.flatten()))
    if np.any(train_data == 0):
        train_data_mean = train_data.flatten()[np.nonzero(train_data.flatten())].mean()
        train_data_std = train_data.flatten()[np.nonzero(train_data.flatten())].std()
        test_data_mean = test_data.flatten()[np.nonzero(test_data.flatten())].mean()
        test_data_std = test_data.flatten()[np.nonzero(test_data.flatten())].std()
        norm_train = (train_data - train_data_mean) / train_data_std
        norm_test = (test_data - test_data_mean) / test_data_std
    else:
        norm_train = (train_data - np.mean(train_data)) / np.std(train_data)
        norm_test = (test_data - np.mean(test_data)) / np.std(test_data)
    return norm_train, norm_test

def get_arr(a, b):
    arr = []
    for i in range(a):
        for j in range(b):
            arr.append([i,j])
    return arr
def DL_arr(a, pre):
    arr = []
    for i in range(a):
        arr.append(pre[i,:,:].flatten())
    return arr
def UL_arr(a, com):
    arr = []
    for i in range(a):
        arr.append(com[:, i, :].flatten())
    return arr

def build_DNN(pre, com, norm_pre, norm_com, AA, MM, norm_AP_dis, norm_MS_dis, norm_AM_dis, DLsub, ULsub):
    n = norm_pre.shape[0]
    # Nt = norm_csi_real.shape[-1]
    n_AP = norm_AP_dis.shape[1]
    n_MS = norm_MS_dis.shape[1]
    data = HeteroData()
    #x1 = np.concatenate((np.array(DL_arr(n_AP, norm_pre)),np.ones((n_AP,1))), axis=1)
    x1 = np.array(DL_arr(n_AP, norm_pre))
    x2 = np.array(UL_arr(n_MS, norm_com))

    # dataAP0=x1.reshape(24,6,4)
    # dataMS = x2.reshape(6,24,2)
    # dataAP=dataAP0.transpose(1,0,2)

    dataAP0=x1.reshape(24,6,4)
    dataMS = x2.reshape(6,24,2)
    dataAP=dataAP0.transpose(1,0,2)

    data= np.concatenate((dataAP,dataMS),axis=2)
    data=data.transpose(2,0,1)
    return data

def build_graph(pre, com, norm_pre, norm_com, AA, MM, norm_AP_dis, norm_MS_dis, norm_AM_dis, DLsub, ULsub):
    n = norm_pre.shape[0]
    # Nt = norm_csi_real.shape[-1]
    n_AP = norm_AP_dis.shape[1]
    n_MS = norm_MS_dis.shape[1]
    data = HeteroData()
    #x1 = np.concatenate((np.array(DL_arr(n_AP, norm_pre)),np.ones((n_AP,1))), axis=1)
    x1 = np.array(DL_arr(n_AP, norm_pre))
    x2 = np.array(UL_arr(n_MS, norm_com))
    data['AP'].x = torch.tensor(x1, dtype=torch.float)
    data['MS'].x = torch.tensor(x2, dtype=torch.float)
    data['AP', 'IAI', 'AP'].edge_index = torch.tensor(np.array(get_arr(n_AP, n_AP)).transpose(), dtype=torch.long)
    data['MS', 'IMI', 'MS'].edge_index = torch.tensor(np.array(get_arr(n_MS, n_MS)).transpose(), dtype=torch.long)
    data['AP', 'DL', 'MS'].edge_index = torch.tensor(np.array(get_arr(n_AP, n_MS)).transpose(), dtype=torch.long)
    data['MS', 'UL', 'AP'].edge_index = torch.tensor(np.array(get_arr(n_MS, n_AP)).transpose(), dtype=torch.long)
    data['AP', 'IAI', 'AP'].edge_attr = torch.tensor(np.expand_dims(norm_AP_dis.flatten(), axis=1), dtype=torch.float)
    data['MS', 'IMI', 'MS'].edge_attr = torch.tensor(np.expand_dims(norm_MS_dis.flatten(), axis=1), dtype=torch.float)
    data['AP', 'DL', 'MS'].edge_attr = torch.tensor(np.expand_dims(norm_AM_dis.flatten(), axis=1), dtype=torch.float)
    norm_AM_dis_T = np.transpose(norm_AM_dis)
    data['MS', 'UL', 'AP'].edge_attr = torch.tensor(np.expand_dims(norm_AM_dis_T.flatten(), axis=1), dtype=torch.float)

    y_pre = []
    y_com = []
    for i in range(n_AP):
        y_pre.append(pre[i, :, :].flatten())
        y_com.append(com[i, :, :].flatten())
    y_pre = torch.tensor(np.array(y_pre), dtype=torch.float)
    y_com = torch.tensor(np.array(y_com), dtype=torch.float)
    data.y_pre = y_pre
    data.y_com = y_com
    data.AA = torch.tensor(AA, dtype=torch.float)
    data.MM = torch.tensor(MM, dtype=torch.float)
    return data


def proc_data(pre, com, norm_pre, norm_com, AA, MM, norm_AP_dis, norm_MS_dis, norm_AM_dis, config, train_layouts):
    #n = norm_pre.shape[0]
    n = train_layouts
    DLsub = config.DLsub
    ULsub = config.ULsub
    data_list = []
    for i in range(n):
        data = build_DNN(pre[i, :, :, :], com[i, :, :, :], norm_pre[i, :, :, :], norm_com[i, :, :, :],
        AA[i, :, :], MM[i, :, :], norm_AP_dis[i, :, :], norm_MS_dis[i, :, :], norm_AM_dis[i, :, :], DLsub, ULsub)
        data_list.append(data)
    return np.array(data_list)

def proc_data_graph(pre, com, norm_pre, norm_com, AA, MM, norm_AP_dis, norm_MS_dis, norm_AM_dis, config, train_layouts):
    #n = norm_pre.shape[0]
    n = train_layouts
    DLsub = config.DLsub
    ULsub = config.ULsub
    data_list = []
    for i in range(n):
        data = build_graph(pre[i, :, :, :], com[i, :, :, :], norm_pre[i, :, :, :], norm_com[i, :, :, :],
        AA[i, :, :], MM[i, :, :], norm_AP_dis[i, :, :], norm_MS_dis[i, :, :], norm_AM_dis[i, :, :], DLsub, ULsub)
        data_list.append(data)
    return data_list


def MDD_loss(AP_pow, MS_pow, config, data):
    #AP_pow = torch.where(AP_pow)
    #MS_pow = MS_pow[:, 0:2]
    AP_mask = torch.reshape(AP_pow, (-1,config.l_AP,AP_pow.size()[1]))
    MS_mask = torch.reshape(MS_pow, (-1, config.d_MS, MS_pow.size()[1]))
    AP_tot = torch.sum(AP_mask, 2, keepdim=True)
    MS_tot = torch.sum(MS_mask, 2, keepdim=True)
    Batchs = AP_tot.size()[0]
    APpow_cons = torch.sum(F.relu(torch.squeeze(AP_tot) - config.AP_power * torch.ones([Batchs, config.l_AP], device=device)))
    MSpow_cons = torch.sum(F.relu(torch.squeeze(MS_tot) - config.MS_power * torch.ones([Batchs, config.d_MS], device=device)))
    MM_mask = torch.reshape(data.MM, (-1,config.d_MS,config.d_MS))
    AA_mask = torch.reshape(data.AA, (-1,config.l_AP,config.l_AP))
    y_pre_mask = torch.reshape(data.y_pre, (-1,config.l_AP,data.y_pre.size()[1]))
    y_com_mask = torch.reshape(data.y_com, (-1, config.l_AP, data.y_com.size()[1]))
    noisePlusinter_MS = torch.squeeze(
        torch.sum((MS_tot * MM_mask), 2, keepdim=True) * config.IMI_SIC / config.Sumsub + MS_tot * config.MS_SIC)+ config.noise_power
    noisePlusinter_AP = torch.squeeze(
    torch.sum((AP_tot * AA_mask), 2, keepdim=True) * config.IAI_SIC / config.Sumsub + AP_tot * config.AP_SIC) + config.noise_power
    DL_signal = torch.squeeze(torch.pow(torch.sum((torch.sqrt(AP_mask) * y_pre_mask), 1, keepdim=True), 2))
    #DL_NI = torch.tensor(np.repeat(noisePlusinter_MS.detach().cpu().numpy(), config.DLsub, axis=1), dtype=float, device=device,requires_grad=True)
    DL_NI = torch.repeat_interleave(noisePlusinter_MS, config.DLsub, dim=1)
    DL_sum = torch.log(1 + torch.div(DL_signal, DL_NI))
    DL_help = torch.eye(config.d_MS, device=device)
    DL_help = torch.repeat_interleave(DL_help, config.DLsub, dim=0)
    DL_MS_sum = DL_sum @ DL_help
    DL_MS_Con = torch.sum(F.relu(config.DL_Con * torch.ones([Batchs, config.d_MS], device=device) - DL_MS_sum))
    temp = torch.squeeze(torch.sum(torch.unsqueeze(noisePlusinter_AP,2) * y_com_mask, 1, keepdim=True))
    UL_NI = torch.reshape(temp, MS_pow.size())
    UL_signal = MS_pow * pow(config.l_AP, 2)
    UL_sum_mask = torch.reshape(torch.log(1 + torch.div(UL_signal, UL_NI)), (DL_sum.size()[0],config.d_MS,-1))
    UL_sum = torch.reshape(UL_sum_mask, (UL_sum_mask.size()[0],-1))
    UL_help = torch.eye(config.d_MS, device=device)
    UL_help = torch.repeat_interleave(UL_help, config.ULsub, dim=0)
    UL_MS_sum = UL_sum @ UL_help
    UL_MS_Con = torch.sum(F.relu(config.UL_Con * torch.ones([Batchs, config.d_MS], device=device) - UL_MS_sum))
    SE_sum = torch.mean(torch.sum(DL_sum,1) + torch.sum(UL_sum,1)) / config.Sumsub
    loss = torch.neg(SE_sum) + 0.1 * APpow_cons + 0.1 * MSpow_cons + 0.1 * DL_MS_Con + 1 * UL_MS_Con
    SE_each=(torch.sum(DL_sum,1) + torch.sum(UL_sum,1)) / config.Sumsub
    return loss, SE_sum,SE_each

train_layouts = 10000
test_layouts = 1000
BatchSize = 64

trainFile = 'C:\\MDD\\WorkData\\train_246_10000.mat'
testFile = 'C:\\MDD\\WorkData\\test_246_1000.mat'
train_data = scio.loadmat(trainFile)
test_data = scio.loadmat(testFile)
train_config = init_parameters()
test_config = init_parameters()

train_channelgain_precoder, train_channelgain_combiner = train_data['f_zf'], train_data['w_zf']
train_AA, train_MM = train_data['AA'], train_data['MM']
train_AP_dis, train_MS_dis, train_AM_dis = train_data['AA_dis'], train_data['MM_dis'], train_data['AM_dis']
test_channelgain_precoder, test_channelgain_combiner = test_data['f_zf'], test_data['w_zf']
test_AA, test_MM = test_data['AA'], test_data['MM']
test_AP_dis, test_MS_dis, test_AM_dis = test_data['AA_dis'], test_data['MM_dis'], test_data['AM_dis']


norm_train_pre, norm_test_pre = normalize_data(train_channelgain_precoder, test_channelgain_precoder)
norm_train_com, norm_test_com = normalize_data(train_channelgain_combiner, test_channelgain_combiner)
norm_train_AP_dis, norm_test_AP_dis = normalize_data(train_AP_dis, test_AP_dis)
norm_train_MS_dis, norm_test_MS_dis = normalize_data(train_MS_dis, test_MS_dis)
norm_train_AM_dis, norm_test_AM_dis = normalize_data(train_AM_dis, test_AM_dis)
norm_train_AA, norm_train_MM = normalize_data(train_AA, train_MM)
norm_test_AA, norm_test_MM = normalize_data(test_AA, test_MM)

# 得到DNN数据格式
train_data_list = proc_data(train_channelgain_precoder, train_channelgain_combiner,
norm_train_pre, norm_train_com, train_AA, train_MM, norm_train_AP_dis, norm_train_MS_dis, norm_train_AM_dis, train_config, train_layouts)
test_data_list = proc_data(test_channelgain_precoder, test_channelgain_combiner,
norm_test_pre, norm_test_com, test_AA, test_MM, norm_test_AP_dis, norm_test_MS_dis, norm_test_AM_dis, test_config, test_layouts)



data = train_data_list[0]
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.ReLU())
        self.module2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)) )
        self.module3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(3, 3), stride=1, padding=(1, 1)), nn.ReLU())
        self.module4 = nn.Sequential(
            nn.Conv2d(48, 60, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.ReLU())
        self.module5 = nn.Sequential(nn.Linear(8640, 1500))
        self.module6 = nn.Sequential(nn.Linear(1500, 588), nn.ReLU())



    def forward(self,x):
        x1=self.module1(x)
        x2=self.module2(x1)
        x3=self.module3(x2)
        x4 = self.module4(x3)
        x=x4

        x=torch.flatten(x,start_dim=2,end_dim=3)
        x=torch.flatten(x,start_dim=1,end_dim=2)
        x = self.module5(x)
        x = self.module6(x)

        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup_seed(100)
model = CNNnet()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

train_data_tensor = TensorDataset(torch.FloatTensor(np.array(train_data_list)))
train_loader = torch.utils.data.DataLoader(train_data_tensor , batch_size=BatchSize, shuffle=True,num_workers=0)
test_data_tensor =TensorDataset(torch.FloatTensor(test_data_list))
test_loader = torch.utils.data.DataLoader(test_data_tensor , batch_size=test_layouts, shuffle=False, num_workers=0)

train_data_list_graph = proc_data_graph(train_channelgain_precoder, train_channelgain_combiner,
norm_train_pre, norm_train_com, train_AA, train_MM, norm_train_AP_dis, norm_train_MS_dis, norm_train_AM_dis, train_config, train_layouts)
test_data_list_graph = proc_data_graph(test_channelgain_precoder, test_channelgain_combiner,
norm_test_pre, norm_test_com, test_AA, test_MM, norm_test_AP_dis, norm_test_MS_dis, norm_test_AM_dis, test_config, test_layouts)
data = train_data_list[0]
train_loader_graph = DataLoader(train_data_list_graph, batch_size=BatchSize, shuffle=True,num_workers=0)
test_loader_graph = DataLoader(test_data_list_graph, batch_size=test_layouts, shuffle=False, num_workers=0)

def train():
    model.train()

    total_loss = 0
    total_SE = 0
    for (data_new, data_graph) in zip(train_loader, train_loader_graph):
        data= data_new[0].to(device)
        data_graph = data_graph.to(device)
        optimizer.zero_grad()
        pow = model(data)
        AP_pow0=pow[:,:576]
        MS_pow0 = pow[:,576:588]
        b_size=pow.shape[0]
        AP_pow=AP_pow0.reshape(b_size,24,24)
        AP_pow=torch.flatten(AP_pow, start_dim=0, end_dim=1)
        MS_pow=MS_pow0.reshape(b_size,6,2)
        MS_pow = torch.flatten(MS_pow, start_dim=0, end_dim=1)
        loss, SE,SE_each = MDD_loss(AP_pow, MS_pow, train_config, data_graph)
        loss.backward()
        total_loss += loss.item() * b_size
        total_SE += SE.item() * b_size
        optimizer.step()
    return total_loss / train_layouts, total_SE / train_layouts,SE_each

def test():
    model.eval()

    total_loss = 0
    total_SE = 0
    for (data_new, data_graph) in zip(test_loader, test_loader_graph):
        data = data_new[0].to(device)
        data_graph = data_graph.to(device)
        with torch.no_grad():
            data = torch.squeeze(data, dim=2)
            pow = model(data)
            AP_pow0 = pow[:, :576]
            MS_pow0 = pow[:, 576:588]
            b_size = pow.shape[0]
            AP_pow = AP_pow0.reshape(b_size, 24, 24)
            AP_pow = torch.flatten(AP_pow, start_dim=0, end_dim=1)
            MS_pow = MS_pow0.reshape(b_size, 6, 2)
            MS_pow = torch.flatten(MS_pow, start_dim=0, end_dim=1)
            loss, SE ,SE_each= MDD_loss(AP_pow, MS_pow, test_config, data_graph)
            total_loss += loss.item() * b_size
            total_SE += SE.item() * b_size
    return total_loss/test_layouts, total_SE/test_layouts,SE_each

val_max = 0
for epoch in range(1, 200):
    loss1, SE,SE_each= train()
    loss2, SE_test,SE_each_test=test()
    print('Epoch {:03d}, Train Loss: {:.4f}, Total SE: {:.4f},Test Loss: {:.4f}, Test SE: {:.4f}'.format(
        epoch, loss1, SE,loss2, SE_test))
    scheduler.step()
    if epoch==1:
        val_max=SE_test
    else:
        if SE_test > val_max:
            val_max =SE_test
            best_model = copy.deepcopy(model)


# save box plot
# model.load_state_dict(torch.load('D:\\LBH\\MDD\\ModelSave\\model_246_CNN'))
# model.eval()
# loss2, SE_test, SE_each_test = test()
# path = 'D:\\LBH\\MDD\\WorkData'
# fname1 = os.path.join(path,'CNN_246_test'+'.h5')
# cachenew(fname1, SE_each_test.cpu().numpy())
