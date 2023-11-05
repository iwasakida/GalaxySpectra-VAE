# torch関連ライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
#from torchinfo import summary
#from torchviz import make_dot
#from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import numpy as np
import time
SEED = 123

# デバイスの割り当て
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train = np.load('...')
test = np.load('...')

train = torch.tensor(train)
test = torch.tensor(test)

torch_seed()
batch_size = 256
num_workers = 0

train_data = TensorDataset( train,train)
trainloader = DataLoader(train_data, batch_size = batch_size, shuffle=True,num_workers=num_workers)

test_data = TensorDataset( test,test)
testloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,num_workers=num_workers)

MSE = torch.nn.MSELoss()
from Model import Spectra_VAE
from Model_train import train_VAE, KL_Loss
for n_latent in [1,2,3,4,5,6,7,8,9,10]:
    model = spectra_VAE(latent_dim = n_latent)
    model = model.to(device)
    modelfile = f'path_model/VAE_model_latent{n_latent}.pth'
    model.load_state_dict(torch.load(modelfile))
    train_VAE(
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
