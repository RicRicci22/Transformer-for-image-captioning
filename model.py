import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import resnet101,resnet152
from pytorch_pretrained_vit import ViT
from torchvision.models._utils import IntermediateLayerGetter
from torchsummary import summary
import sys
torch.set_printoptions(threshold=sys.maxsize)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, heads):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        
        assert (self.head_dim*heads) == hidden_size, "Hidden size must be divisible by heads"
        # Define the three linear transformation of the embeddings 
        self.linear_query = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.linear_key = nn.Linear(self.head_dim, self.head_dim,bias=False)
        self.linear_values = nn.Linear(self.head_dim, self.head_dim,bias=False)
        
        self.fc_out = nn.Linear(hidden_size,hidden_size)
    
    def forward(self,queries,keys,values,mask=None):
        # Get information for multi head attention
        BATCH_SIZE = queries.shape[0]
        value_len,key_len,query_len = values.shape[1],keys.shape[1],queries.shape[1]
        
        values = values.reshape((BATCH_SIZE,value_len,self.heads,self.head_dim))
        keys = keys.reshape((BATCH_SIZE,key_len,self.heads,self.head_dim))
        queries = queries.reshape((BATCH_SIZE,query_len,self.heads,self.head_dim))
        
        queries = self.linear_query(queries)
        values = self.linear_values(values)
        keys = self.linear_key(keys)
        
        weights = torch.einsum('nqhd,nkhd->nhqk',[queries,keys])
        if mask is not None:
            weights = weights.masked_fill(mask==0,float('-1e20'))
        
        weights = F.softmax(weights/(self.hidden_size**(1/2)), dim=3)

        # weights [BATCH_SIZE, heads, LEN, LEN]
        # values [BATCH_SIZE,LEN,heads,head_dim]
        # out [BATCH_SIZE,LEN,heads,head_dim]
        
        out = torch.einsum('nhqk,nkhd->nqhd',[weights,values]).reshape(BATCH_SIZE,query_len,self.heads*self.head_dim)
        
        out = self.fc_out(out)
        return out 

class Transformer_block(nn.Module):
    def __init__(self, hidden_size, heads, dropout, forward_expansion):
        super(Transformer_block,self).__init__()
        self.attention = SelfAttention(hidden_size,heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size,forward_expansion*hidden_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*hidden_size,hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, mask=None):
        attention = self.attention(queries,keys,values,mask)
        x = self.dropout(self.norm1(attention+queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out
    

class Encoder(nn.Module):
    def __init__(self,hidden_size,num_layers,heads,device,forward_exp,dropout,image_position=64):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.position_embed = nn.Embedding(image_position,hidden_size)
        
        self.linear_proj = nn.Linear(2048,hidden_size)
        
        self.layers = nn.ModuleList([
            Transformer_block(hidden_size,heads,dropout,forward_exp)
            ]*num_layers)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        N, LEN, _= x.shape
        positions = torch.arange(0,LEN).expand(N,LEN).to(self.device) # Crea una matrice di dimensione N,LEN con numeri progressivi in ogni riga 
        
        out = self.dropout(self.linear_proj(x) + self.position_embed(positions)) # Need positional embeedding

        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out 


class DecoderBlock(nn.Module):
    def __init__(self,hidden_size,heads,forward_exp,dropout):
        super(DecoderBlock,self).__init__()
        self.attention = SelfAttention(hidden_size,heads)
        self.norm = nn.LayerNorm(hidden_size)
        self.transformer_block = Transformer_block(hidden_size,heads,dropout,forward_exp)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x, keys, values,trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        queries = self.dropout(self.norm(attention+x))
        out = self.transformer_block(queries,keys,values)
        return out

class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,hidden_size,num_layers,heads,forward_exp,dropout,device,max_len):
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size,hidden_size,padding_idx=0)
        self.position_embedding = nn.Embedding(max_len,hidden_size)
        
        self.layers = nn.ModuleList(
            [DecoderBlock(hidden_size,heads,forward_exp,dropout)
             ]*num_layers
        )
        self.fc = nn.Linear(hidden_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,enc_out,trg_mask):
        N, LEN = x.shape
        positions = torch.arange(0,LEN).expand(N,LEN).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out,enc_out,enc_out,trg_mask)
        
        out = self.fc(out)
        
        return out

class ResnetBackbone(nn.Module):
    def __init__(self,pretrained: bool,trainable: bool):
        super(ResnetBackbone,self).__init__()
        backbone = resnet152(pretrained=pretrained)
        return_layers = {'layer4': "0"}
        # Pretraining as in catr! 
        for name , parameter in backbone.named_parameters():
            if not trainable or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
    
    def forward(self, x):
        # x is a tensor of shape [N, C, H, W] 
        xs = self.body(x)
        out = xs['0']
        out = out.view(out.shape[0],-1,out.shape[1])
        return out

class ViTBackbone(nn.Module):
    def __init__(self,pretrained: bool,trainable: bool):
        super(ViTBackbone,self).__init__()
        self.backbone = ViT('B_16_imagenet1k',pretrained=pretrained,image_size=256)
        #summary(backbone, input_size=(3,256,256),device='cpu')
        # for name, param in backbone.named_parameters():
        #     print(name, param.shape)
        return_layers = {'norm': "0"}
        # Pretraining as in catr! 
        for _, parameter in self.backbone.named_parameters():
            # if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            if not trainable:
                parameter.requires_grad_(False)
        
        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)
    
    def forward(self, x):
        # x is a tensor of shape [N, C, H, W] 
        x = self.backbone(x)
        # xs = self.body(x)
        # out = xs['0']
        # out = out.reshape(out.shape[0],-1,out.shape[1])
        # return out
        return x[:,1:,:]
        

class Transformer(nn.Module):
    def __init__(self,trg_vocab_size,trg_pad_idx,hidden_size=256,
                 num_layers=6,forward_exp=4,heads=8,dropout=0,device='cuda',max_length=100, backbone='resnet',pretrained = True, train_backbone=True):
        super(Transformer,self).__init__()
        self.backbone_type = backbone
        self.decoder = Decoder(trg_vocab_size,hidden_size,num_layers,heads,forward_exp,dropout,device,max_length)
        if(self.backbone_type == 'resnet'):
            self.backbone = ResnetBackbone(pretrained=pretrained,trainable=train_backbone)
            self.encoder = Encoder(hidden_size,num_layers,heads,device,forward_exp,dropout)
        elif(self.backbone_type == 'vit'):
            self.backbone = ViTBackbone(pretrained=pretrained,trainable=train_backbone)
            self.linear_proj = nn.Linear(768,hidden_size,bias=False)
            
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.max_len = max_length
        self.hidden_size = hidden_size
    
    def forward(self,src,trg,trg_mask):
        if(self.backbone_type=='resnet'):
            src = self.backbone(src) # Backbone prende una batch di dimensione (N,C,W,H) in ingresso e restituisce un tensore di dimensione (N,LEN,HIDDEN_SIZE)
            enc_src = self.encoder(src,None)
        elif(self.backbone_type=='vit'):
            enc_src = self.backbone(src)
            enc_src = self.linear_proj(enc_src)
        else:
            print('Backbone not supported')
            
        out = self.decoder(trg,enc_src,trg_mask)
        return out 
    

# CHECK OUT THIS https://github.com/wtliao/ImageTransformer