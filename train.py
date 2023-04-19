import math
import torch
import torch.nn as nn
import os
import datetime
from networkTool import *
from torch.utils.tensorboard import SummaryWriter
from models.attentionModel import TransformerLayer,TransformerModule
import dataset
import torch.utils.data as data
import time
import os
import hydra
from omegaconf import DictConfig, OmegaConf


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone()
    target = source[i+1:i+1+seq_len,:,-1,0].reshape(-1)
    data[:,:,0:-1,:] = source[i+1:i+seq_len+1,:,0:-1,:] # this moves the feat(octant,level) of current node to lastrow,        
    data[:,:,-1,1:3] = source[i+1:i+seq_len+1,:,-1,1:3]# which will be used as known feat
    return data[:,:,-levelNumK:,:], (target).long(),[]



######################################################################
# Run the model
# -------------
#
model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
if __name__=="__main__":

    epochs = 8 # The number of epochs
    best_model = None
    batch_size = 128
    TreePoint = bptt*16
    train_set = dataset.DataFolder(root=trainDataRoot, TreePoint=TreePoint,transform=None,dataLenPerFile= 391563.61670395226) # you should run 'dataLenPerFile' in dataset.py to get this num (17456051.4)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=True) # will load TreePoint*batch_size at one time
    
    # loger
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    printl = CPrintl(expName+'/loss.log')
    writer = SummaryWriter('./log/'+expName)
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    # model_structure(model,printl)
    printl(expComment+' Pid: '+str(os.getpid()))
    log_interval = int(batch_size*TreePoint/batchSize/bptt)
    
    # learning
    criterion = nn.CrossEntropyLoss()
    lr = 1e-3 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    idloss = 0

    # reload
    saveDic = None
    # saveDic = reload(100030,checkpointPath)
    if saveDic:
        scheduler.last_epoch = saveDic['epoch'] - 1
        idloss = saveDic['idloss']
        best_val_loss = saveDic['best_val_loss']
        model.load_state_dict(saveDic['encoder'])
        
    def train(epoch):
        global idloss,best_val_loss
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        total_loss_list = torch.zeros((1,7))
            
        for Batch, d in enumerate(train_loader): # there are two 'BATCH', 'Batch' includes batch_size*TreePoint/batchSize/bptt 'batch'es.
            batch = 0
 
            train_data = d[0].reshape((batchSize,-1,4,6)).to(device).permute(1,0,2,3)   #shape [TreePoint*batch_size(data)/batch_size,batch_size,7,6]
            src_mask = model.generate_square_subsequent_mask(bptt).to(device)
            for index, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
                data, targets,dataFeat = get_batch(train_data, i)#data [35,20]
                optimizer.zero_grad()
                if data.size(0) != bptt:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = model(data, src_mask,dataFeat)                         #output: [bptt,batch size,255]
                loss = criterion(output.view(-1, ntokens), targets)/math.log(2)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                batch = batch+1

                if batch % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                
                    total_loss_list = " - "
                    printl('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | '
                        'lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | losslist  {} | ppl {:8.2f}'.format(
                            epoch, Batch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss,total_loss_list, math.exp(cur_loss)))
                    total_loss = 0
    
                    start_time = time.time()

                    writer.add_scalar('train_loss', cur_loss,idloss)
                    idloss+=1

            if Batch%10==0:
                save(epoch*100000+Batch,saveDict={'encoder':model.state_dict(),'idloss':idloss,'epoch':epoch,'best_val_loss':best_val_loss},modelDir=checkpointPath)
    
    # train
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)

@hydra.main(version_base=None, config_path="config", config_name="obj")
def main(cfg : DictConfig) -> None:
    pass

if __name__ == '__main__':
    main()


# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split
# from torchvision.datasets import MNIST
# from torchvision import transforms
# import pytorch_lightning as pl

# class LitAutoEncoder(pl.LightningModule):
# 	def __init__(self):
# 		super().__init__()
# 		self.encoder = nn.Sequential(
#       nn.Linear(28 * 28, 64),
#       nn.ReLU(),
#       nn.Linear(64, 3))
# 		self.decoder = nn.Sequential(
#       nn.Linear(3, 64),
#       nn.ReLU(),
#       nn.Linear(64, 28 * 28))

# 	def forward(self, x):
# 		embedding = self.encoder(x)
# 		return embedding

# 	def configure_optimizers(self):
# 		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
# 		return optimizer

# 	def training_step(self, train_batch, batch_idx):
# 		x, y = train_batch
# 		x = x.view(x.size(0), -1)
# 		z = self.encoder(x)    
# 		x_hat = self.decoder(z)
# 		loss = F.mse_loss(x_hat, x)
# 		self.log('train_loss', loss)
# 		return loss

# 	def validation_step(self, val_batch, batch_idx):
# 		x, y = val_batch
# 		x = x.view(x.size(0), -1)
# 		z = self.encoder(x)
# 		x_hat = self.decoder(z)
# 		loss = F.mse_loss(x_hat, x)
# 		self.log('val_loss', loss)

# # data
# dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
# mnist_train, mnist_val = random_split(dataset, [55000, 5000])

# train_loader = DataLoader(mnist_train, batch_size=32)
# val_loader = DataLoader(mnist_val, batch_size=32)

# # model
# model = LitAutoEncoder()

# # training
# trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
# trainer.fit(model, train_loader, val_loader)
    