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
import shap
SEED = 123

# デバイスの割り当て
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[4]:


# PyTorch乱数固定用
def torch_seed(seed=123):
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

model = load_model(n_latent).to(device)
modelfile =  'model_hpc/model latent{} beta01.pth'.format(n_latent)
model.load_state_dict(torch.load(modelfile, map_location=torch.device(device)))

train = np.load('train.npy')
test = np.load('test.npy')

train = torch.tensor(train)
test = torch.tensor(test)

train_data = TensorDataset( train )
test_data = TensorDataset( test )

trainloader = DataLoader(train_data, batch_size = 32, shuffle=False,num_workers=0)
testloader = DataLoader(test_data, batch_size = 32, shuffle=False,num_workers=0)

torch_seed()
n_latent = 3
latent_train = np.zeros( (  len(train), n_latent ) )
base = 0
model.eval()
with torch.no_grad():
    for i, (inputs) in enumerate(trainloader):

        outputs, mean, log_var, z = model(inputs.float())

        latent_train[base:base+len(inputs)] = mean.numpy()
        base += len(inputs)

latent = np.zeros( (  len(test), n_latent ) )
base = 0
model.eval()
with torch.no_grad():
    for i, (inputs) in enumerate(testloader):

        outputs, mean, log_var, z = model(inputs.float())

        latent[base:base+len(inputs)] = mean.numpy()
        base += len(inputs)

print(latent[[0,40,10000,-1,-10]])
print(latent_train[[0,40,10000,-1,-10]])


from sklearn.decomposition import PCA
Pca_latent = PCA(n_components=n_latent,random_state=SEED)
pca_latent_train = Pca_latent.fit_transform(latent_train)
pca_latent = Pca_latent.transform(latent)
Mean_latent = torch.tensor( np.mean(latent_train, axis=0))
PCA_component = torch.tensor( Pca_latent.components_ )

def load_model_encoder(n_latent):
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

    class VAE(torch.nn.Module):
        def __init__(self, Encoder, Mean):
            super().__init__()
            self.encoder = Encoder
            self.mean = Mean

        def reparameterize(self, z_mean, z_log_var):
            epsilon = torch.randn(z_mean.shape).to(device)
            z = z_mean + torch.exp(z_log_var / 2) * epsilon
            return z

        def forward(self, x):

            x = self.encoder(x).to(device)

            #平均 μ
            mean = self.mean(x).to(device)

            mean = mean - Mean_latent

            mean = torch.mm( mean, PCA_component.t() ).to(device)
            return mean

    model = VAE(encoder, Mean).to(device)

    return model


model = load_model(n_latent)
modelfile =  'model_hpc/model latent{} beta01.pth'.format(n_latent)
model.load_state_dict(torch.load(modelfile, map_location=torch.device(device)))
Encoder = load_model_encoder(n_latent)
Encoder.encoder.load_state_dict(model.encoder.state_dict())
Encoder.mean.load_state_dict(model.mean.state_dict())


train = np.load('train.npy')
test = np.load('test.npy')
train_M = np.load('log_M_cal_train.npy')
test_M = np.load('log_M_cal_train.npy')

bins = 5
label = 'log_M*_cal'
Range = np.linspace(8.5,12,5)
n_list_train = np.zeros((0,4000))
for i in range(bins):
    if i == 0:
        criteria = ( train_M < Range[i]  )
    else:
        criteria = ( Range[i-1] < train_M ) & ( Range[i] > train_M )

    sorting_indices = np.argsort( train_M[criteria] )
    n_list_train = np.vstack( [n_list_train,train[sorting_indices[:2500]]])


n_list_test = np.zeros((0,4000))
for i in range(bins):
    if i == 0:
        criteria = ( test_M < Range[i]  )
    else:
        criteria = ( Range[i-1] < test_M   ) & ( Range[i] > test_M  )

    sorting_indices = np.argsort(test_M[criteria])
    n_list_test = np.vstack( [n_list_test,test[sorting_indices[:50]]])

import shap
background = torch.tensor(
    n_list_train
).float()

explainer = shap.DeepExplainer(Encoder, background)

model.eval()
torch_seed()
# Calculate the SHAP values for the test instance
shap_values = explainer.shap_values(torch.tensor(n_list_test).float())
np.save('SHAP vlaue VAE3.npy',shap_values)
np.save('VAE3 beta01 SHAP values spectra.npy',n_list_test)