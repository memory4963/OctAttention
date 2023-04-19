import math
import torch
from torch import nn
import pytorch_lightning as pl

from models.attentionModel import TransformerLayer,TransformerModule


class OctAttention(pl.LightningModule):
    def __init__(self, cfg):
        super(OctAttention, self).__init__()
        self.cfg = cfg

        self.pos_encoder = PositionalEncoding(cfg.model.embed_dimension, cfg.train.dropout)

        encoder_layers = TransformerLayer(cfg.model.embed_dimension, cfg.model.head_num, cfg.model.hidden_dimension, cfg.train.dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, cfg.model.layer_num)

        self.encoder = nn.Embedding(cfg.model.token_num, 128)
        
        self.encoder1 = nn.Embedding(cfg.model.max_octree_level+1, 6)
        self.encoder2 = nn.Embedding(9, 4)

        self.cfg.model.embed_dimension =cfg.model.embed_dimension 
        self.act = nn.ReLU()
        self.decoder0 = nn.Linear(cfg.model.embed_dimension, cfg.model.embed_dimension)
        self.decoder1 = nn.Linear(cfg.model.embed_dimension, cfg.model.token_num)

        self.criterion = nn.CrossEntropyLoss()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask, dataFeat):
        bptt = src.shape[0]
        batch = src.shape[1]

        oct = src[:,:,:,0] #oct[bptt,batchsize,FeatDim(levels)] [0~254]
        level = src[:,:,:,1]  # [0~12] 0 for padding data
        octant = src[:,:,:,2] # [0~8] 0 for padding data

        # assert oct.min()>=0 and oct.max()<255
        # assert level.min()>=0 and level.max()<=12
        # assert octant.min()>=0 and octant.max()<=8
        
        level -= torch.clip(level[:,:,-1:] - 10,0,None)# the max level in traning dataset is 10
        torch.clip_(level,0, self.cfg.model.max_octree_level)
        aOct = self.encoder(oct.long()) #a[bptt,batchsize,FeatDim(levels),EmbeddingDim]
        aLevel = self.encoder1(level.long())
        aOctant = self.encoder2(octant.long())

        a = torch.cat((aOct,aLevel,aOctant),3)

        a = a.reshape((bptt,batch,-1)) 
        
        # src = self.ancestor_attention(a)
        src = a.reshape((bptt,a.shape[1],self.cfg.model.embed_dimension))* math.sqrt(self.cfg.model.embed_dimension)

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder1(self.act(self.decoder0(output)))
        return output

    def configure_optimizers(self):
        # just let lightning handle it, change to manual optimization if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
        return optimizer

    def training_step(self, data, targets, dataFeat):
        loss = 0.
        if data.shape[0] != self.cfg.model.context_size:
            src_mask = self.generate_square_subsequent_mask(data.shape[0]).to(self.device)
        pred = self(data, src_mask, dataFeat)
        loss = self.criterion(pred.view(-1, self.cfg.model.token_nums), targets) / math.log(2)
        return loss


class PositionalEncoding(nn.Module):
    def __init__(self, cfg):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=cfg.train.dropout)

        pe = torch.zeros(cfg.model.pos_max_len, cfg.model.embed_dimension)
        position = torch.arange(0, cfg.model.pos_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, cfg.model.embed_dimension, 2).float() * (-math.log(10000.0) / cfg.model.embed_dimension))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
