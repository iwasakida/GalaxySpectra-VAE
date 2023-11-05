# torch関連ライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
import time
SEED = 123

from Model import Spectra_CVAE
from Model_train import train_CVAE, KL_Loss

MSE = nn.MSELoss()

train = np.load(...)
test = np.load(...)
train = torch.tensor(train)
test = torch.tensor(test)
n_class = 1
label_list = [
    'M','SFR','sSFR','Metal',
]
for label in label_list:
    test_label = np.load('path/{}_test.npy'.format(label))
    train_label = np.load('path/{}_train.npy'.format(label))
    test_label = torch.tensor(test_label).reshape(-1,1)
    train_label = torch.tensor(train_label).reshape(-1,1)
    train_data = TensorDataset( train, train_label)
    test_data = TensorDataset( test, test_label)
    trainloader = DataLoader(train_data, batch_size = 256, shuffle=True,num_workers=0)
    testloader = DataLoader(test_data, batch_size = 256, shuffle=False,num_workers=0)
    for n_latent in [1,2,3,4,5,6,7,8,9,10]:
        model =  Spectra_CVAE( latent_dim=n_latent, n_class=n_class )
        model = model.to(device)
        modelfile =  f'path_model/model CVAE latent{n_latent} {label} beta01.pth'
        model.load_state_dict(torch.load(modelfile))
        train_CVAE(
            model=model,
            modelfile=modelfile,
            testloader=testloader,
            trainloader=trainloader,
            n_latent=n_latent,
            recon_loss_func=MSE,
            kl_loss_func=KL_Loss,
            patience=100,
            lr = 5e-4,
            alpha = 1e3,
            beta = 1e-1,
            SEED = SEED,
        )