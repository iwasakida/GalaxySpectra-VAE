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

# PyTorch乱数固定用
def torch_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def model_VAE(n_latent):
    class Reshape(torch.nn.Module):
        def forward(self, x):
            return x.view((-1,1,4000))
    class Flatten(torch.nn.Module):
        def forward(self, x):
            return x.view(-1, int(x.shape[1]*x.shape[2]))
    class Unflatten(torch.nn.Module):
        def forward(self, x):
            return x.view((-1,64,250))
    class Reshape2(torch.nn.Module):
        def forward(self, x):
            return x.view((-1,4000))
    encoder = torch.nn.Sequential( 

        Reshape(),

        nn.Conv1d(1,8,5,padding=1,stride=1),
        nn.BatchNorm1d(8),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.MaxPool1d(2),

        nn.Conv1d(8,16,5,padding=1,stride=1),
        nn.BatchNorm1d(16),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.MaxPool1d(2),

        nn.Conv1d(16,32,5,padding=1,stride=1),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.MaxPool1d(2),

        nn.Conv1d(32,64,5,padding=1,stride=1),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.MaxPool1d(2),

        Flatten(),

        nn.Linear(64*248, 1024),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

    )

    Mean = torch.nn.Sequential(

        nn.Linear(1024, n_latent),

    )

    Log_var = torch.nn.Sequential(

        nn.Linear(1024, n_latent),

    )

    decoder = torch.nn.Sequential(

        nn.Linear(n_latent, 1024),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.Linear(1024, 64*250),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        Unflatten(),

        nn.ConvTranspose1d(64,32,5,stride=2,padding=1,output_padding=1),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.ConvTranspose1d(32,16,5,stride=2,padding=1,output_padding=1),
        nn.BatchNorm1d(16),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.ConvTranspose1d(16,8,5,stride=2,padding=1,output_padding=1),
        nn.BatchNorm1d(8),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.ConvTranspose1d(8,8,3,stride=2,padding=1,output_padding=1),
        nn.BatchNorm1d(8),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(0),

        nn.Conv1d(8,1,3,padding=1,stride=1),
        nn.ReLU(inplace=True),
        Flatten(),

        nn.Linear(4028,4000),
        nn.ReLU(inplace=True),
    )

    class VAE(torch.nn.Module):
        def __init__(self, Encoder, Mean, Log_var, Decoder):
            super().__init__()
            self.encoder = Encoder
            self.mean = Mean 
            self.log_var = Log_var 
            self.decoder = Decoder

        def reparameterize(self, z_mean, z_log_var):
            epsilon = torch.randn(z_mean.shape).to(device)
            z = z_mean + torch.exp(z_log_var / 2) * epsilon
            return z

        def forward(self, x):

            x = self.encoder(x).to(device)

            #平均 μ
            mean = self.mean(x).to(device)
            # log σ^2f
            log_var = self.log_var(x).to(device)

            z = self.reparameterize(mean,log_var).to(device)

            x = self.decoder(z).to(device)

            return x, mean, log_var, z
    model = VAE(encoder, Mean, Log_var,decoder)
    return model

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        # The example code divides by (dim) here, making <kernel_input> ~ 1/dim
        # excluding (dim) makes <kernel_input> ~ 1
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)#/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(self, x, y):
        xx_kernel = compute_kernel(x,x)
        yy_kernel = compute_kernel(y,y)
        xy_kernel = compute_kernel(x,y)
        return torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2*torch.mean(xy_kernel)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

def KL_Loss(mean, log_var):
    # KLダイバージェンス計算
    kl_loss = - 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp()) 
    return kl_loss

# 損失関数はなにがいい
MSE = torch.nn.MSELoss()
#lossf = torch.nn.GaussianNLLLoss()
def Reconstruction_Loss(inputs, outputs,w):
    MSE = torch.mean(w*(inputs-outputs).pow(2))
    return MSE


train = np.load('../data/train.npy')
test = np.load('../data/test.npy')

train_error = np.load('../data/error.npy')
test_error = np.load('../data/error_test.npy')

train_error[np.isnan(train_error)] = 0
test_error[np.isnan(test_error)] = 0

test_sigma = test_error**2 + np.std(test,axis=0)**2
train_sigma = train_error**2 + np.std(train,axis=0)**2

test_sigma = 1 / test_sigma
train_sigma = 1 / train_sigma

train = torch.tensor(train)
test = torch.tensor(test)

log_M_train = np.load('../data/log_M_cal_train.npy')
log_M_test = np.load('../data/log_M_cal_test.npy')

log_M_test[np.isnan(log_M_test)] = np.nan
log_M_test[(log_M_test<7) | (log_M_test>13)] = np.nan

log_M_train[np.isnan(log_M_train)] = np.nan
log_M_train[(log_M_train<7) | (log_M_train>13)] = np.nan

weight_test_cri = np.isnan(log_M_test)
weight_train_cri = np.isnan(log_M_train)

log_M_test = log_M_test[~np.isnan(log_M_test)]
log_M_train = log_M_train[~np.isnan(log_M_train)]

log_M = np.hstack( [ log_M_train, log_M_test ] )

bins = 10
hist_values, bin_edges = np.histogram(log_M, bins=bins)
weights = np.zeros_like(log_M)
for i in range(len(bin_edges)-1):
    criteria = ( bin_edges[i] <= log_M ) & (bin_edges[i+1] >= log_M )
    weights[criteria] = 1/hist_values[i]

weights = torch.tensor(weights)

weights_train = weights[:len(log_M_train)]
weights_test = weights[len(log_M_train):]

train_sigma = train_sigma[~weight_train_cri]
test_sigma = test_sigma[~weight_test_cri]
train = train[~weight_train_cri]
test = test[~weight_test_cri]

train = torch.tensor(train)
test = torch.tensor(test)
train_sigma = torch.tensor(train_sigma)
test_sigma = torch.tensor(test_sigma)

train_data = TensorDataset( train, train_sigma )
test_data = TensorDataset( test, test_sigma )
num_workers = 0


nround = 200
torch_seed()
alpha= 1 - 1e-2
lam = 2
beta =0.1
mmd_weight = 6.72
gamma = 100
a = 1e3
history = np.zeros((0,7))
from sklearn.utils import resample


def train_model(
    model, modelfile,batch_size = 64, patience=20,
    lr = 5e-4,
    one_train_ratio = 50,
    restart = False,
    beta = beta,
):
    nround=20
    torch_seed()
    num_workers = 0

    num_samples=len(weights_train)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_train, num_samples=num_samples)
    trainloader = DataLoader(dataset=train_data,batch_size=batch_size, shuffle=False, drop_last=True,sampler = sampler, pin_memory=False, num_workers=0)

    testloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    history = np.zeros((0,8))
    no_improvement = 0
    val_loss_history = torch.zeros(3)
    with torch.no_grad():
        model.eval()
        for i, ( inputs, weight ) in enumerate(testloader):

            inputs = inputs.to(device).float()
            weight = weight.to(device).float()
            optimizer.zero_grad()

            outputs, mean, log_var, z = model(inputs)

            recon_loss = MSE(inputs,outputs.float())
            kl_loss = KL_Loss(mean, log_var)

            loss = a * recon_loss + beta * kl_loss

            val_loss_history[0] += loss.detach().item()
            val_loss_history[1] += recon_loss.detach().item()
            val_loss_history[2] += kl_loss.detach().item()

        val_loss_history /= i
        best_loss = val_loss_history[0]

    if restart:
        best_loss = 1e10
    for epoch in range(0, 10000000):

        model.train()
        train_loss_history = torch.zeros(4)
        st = time.time()
        for i, ( inputs, weight ) in enumerate(trainloader):

            inputs = inputs.to(device).float()
            weight = weight.to(device).float()
            optimizer.zero_grad()

            outputs, mean, log_var, z = model(inputs)
            recon_loss = MSE(inputs,outputs.float())
            kl_loss = KL_Loss(mean, log_var)

            loss = a * recon_loss + beta * kl_loss
            loss = loss.to(device)

            if torch.any( torch.isnan( loss ) ) or torch.any( torch.isinf( loss ) ):
                model.load_state_dict(torch.load(modelfile))
                print('train loss nan')
                break

            loss.backward()
            optimizer.step()

            train_loss_history[0] += loss.detach().item()
            train_loss_history[1] += recon_loss.detach().item()
            train_loss_history[2] += kl_loss.detach().item()
            train_loss_history[3] += beta

            if i*batch_size >len(train) / one_train_ratio:
                break
        train_loss_history /= i

        model.eval()
        val_loss_history = torch.zeros(3)
        with torch.no_grad():
            for i, ( inputs, weight ) in enumerate(testloader):

                inputs = inputs.to(device).float()
                weight = weight.to(device).float()
                optimizer.zero_grad()

                outputs, mean, log_var, z = model(inputs.float())

                recon_loss = MSE(inputs,outputs.float())
                kl_loss = KL_Loss(mean, log_var)

                loss = a * recon_loss + beta * kl_loss
                loss = loss.to(device)


                if torch.any( torch.isnan( loss ) ) or torch.any( torch.isinf( loss ) ):
                    model.load_state_dict(torch.load(modelfile))
                    print('test loss nan')

                val_loss_history[0] += loss.detach().item()
                val_loss_history[1] += recon_loss.detach().item()
                val_loss_history[2] += kl_loss.detach().item()

            val_loss_history /= i

        if best_loss > val_loss_history[0]:
            print('renew')
            best_loss =  val_loss_history[0]
            torch.save(model.state_dict(), modelfile)

            no_improvement = 0

        else:
            no_improvement += 1
            if no_improvement == patience:
                print('Validation loss has not improved for {} epochs, stopping early.'.format(patience))
                break

        et = time.time()

        item = np.hstack([epoch, train_loss_history.numpy(), val_loss_history.numpy()])
        history = np.vstack([history,item])

        print(
            modelfile,'\n',
            'no_improvement', no_improvement ,'\n',

            'epoch', history[-1][0] ,'\n',

            'loss:', np.round( history[-1][1], nround),'\n',
            'val loss:', np.round( history[-1][1+len(train_loss_history)], nround),'\n',

            'reconstruct loss:', np.round( history[-1][2], nround),'\n',
            'reconstruct  val loss:', np.round( history[-1][2+len(train_loss_history)] ,nround),'\n',

            'KL loss:',  np.round( history[-1][3], nround),'\n',
            'KL val loss:',  np.round( history[-1][3+len(train_loss_history)], nround),'\n',

            'Beta :',history[-1][4], '\n',


            '{}'.format(np.round( (et-st)/60, 5 )),

        )

for n_latent in [6]:
    model = model_VAE(3)
    model = model.to(device)
    modelfile = 'model_hpc/model weighted latent{} beta01.pth'.format(3)
    model.load_state_dict(torch.load(modelfile))
    model.mean[0] = nn.Linear(in_features=1024, out_features=n_latent, bias=True)
    model.log_var[0] = nn.Linear(in_features=1024, out_features=n_latent, bias=True)
    model.decoder[0] = nn.Linear(in_features=n_latent, out_features=1024, bias=True)
    model = model.to(device)
    history = train_model(
        model,
        modelfile = 'model_hpc/model weighted latent{} beta01.pth'.format(n_latent),
        batch_size =32,
        patience=1000,
        one_train_ratio=5,
        lr = 1e-3,
        restart = False
    )

