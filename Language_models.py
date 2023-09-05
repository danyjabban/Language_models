import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def token_xent_sum_loss(logits, tgts):
    """
    logits - bsz x T x V
    tgts - bsz x T
    returns:
      a scalar loss with type torch.Tensor
    """    
    logits = torch.permute(logits, (0,2,1))
    loss = F.cross_entropy(logits, tgts, reduction='sum')
    return loss
    


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mod_dim % config.nhead == 0
        self.attn_lin = nn.Linear( # computes key, query, value projections for all heads, but in batch
            config.mod_dim, 3 * config.mod_dim, bias=config.bias)
        self.out_lin = nn.Linear( # projection to re-combine heads' values
            config.mod_dim, config.mod_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.nhead = config.nhead
        self.mod_dim = config.mod_dim
        mask = ~torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0)) # 1 x 1 x len x len

    def get_queries_keys_values(self, X):
        """
        X - bsz x T x model_dim
        returns:
          Q, K, V, each bsz x T x nhead x model_dim/nhead
        """
        bsz, T, dim = X.size()

        # calculate query, key, values for all heads in parallel
        Q, K, V  = self.attn_lin(X).split( # splits into 3 bsz x T x mod_dim matrices, along 3rd dim
            self.mod_dim, dim=2)

        # form bsz x T x nhead x dim/nhead -> bsz x nhead x T x dim/nhead views of each matrix
        Q = Q.view(bsz, T, self.nhead, self.mod_dim // self.nhead).transpose(1, 2)
        K = K.view(bsz, T, self.nhead, self.mod_dim // self.nhead).transpose(1, 2)
        V = V.view(bsz, T, self.nhead, self.mod_dim // self.nhead).transpose(1, 2)
        return Q, K, V

    
    def compute_causal_attn_matrices(self, Q, K):
        """
        Q - bsz x nhead x T x head_dim
        K - bsz x nhead x T x head_dim
        returns:
          bsz x nhead x T x T attention tensor
        """
        att = torch.matmul(Q, K.permute(0,1,3,2))
        
        # mask attention
        m = torch.ones(att.shape) 
        m = torch.tril(m)
        att = att.masked_fill(m == 0, -torch.inf) 

        # initialize softmax, apply on each row
        s = nn.Softmax(dim=3) 
        
        # scale attention
        scale = 1/torch.sqrt(torch.tensor(self.mod_dim))
        att = scale * att

        # apply softmax
        att = s(att)

        att = self.attn_dropout(att) # bsz x nhead x T x T
        
        return att
    
    
    def forward(self, X):
        """
        X - bsz x T x model_dim
        returns:
          bsz x T x V logits tensor
        """
        bsz, T, dim = X.size() # batch size, sequence length, input embedding dim
        Q, K, V = self.get_queries_keys_values(X)
        att = self.compute_causal_attn_matrices(Q, K) # bsz x nhead x T x T
        
        # get attention
        Z = torch.matmul(att,V)
        
        # transpose to concat heads on head-dim
        Z = Z.permute(0, 2, 1, 3)

        # concatinate heads
        Z = Z.contiguous().view(bsz, T, -1)

        # apply Wo transformation
        Z = self.out_lin(Z)
        
        return self.out_dropout(Z)


class CausalConvSelfAttention(CausalSelfAttention):
    def __init__(self, conv, config):
        """
        conv - an nn.Conv1d module
        """
        super().__init__(config)
        assert conv.bias is None
        self.kW = conv.kernel_size[0]
        ctx_len = config.context_length

        self.attn_lin = nn.Linear( # computes value projection for all heads, but in batch
            config.mod_dim , self.kW * config.mod_dim, bias=conv.bias)
        bsz = 2
        self.out_lin_stacked_I = Wo_mat(config.mod_dim, bsz, self.kW)
    
    def get_queries_keys_values(self, X):
        bsz, T, dim = X.size()

        # calculate query, key, values for all heads in parallel
        V  = self.attn_lin(X)

        # form bsz x T x (dim x kW) -> bsz x nhead x T x dim views of matrix V
        V = V.view(bsz, T, self.kW, self.mod_dim * self.kW // self.kW).transpose(1, 2)

        return V

    def compute_causal_attn_matrices(self, T, bsz):
        att = torch.tensor([])
        for i in range(self.kW):
            diag = torch.tensor([1]*(T - i)) # create 1s of correct length
            head = torch.diag_embed(diag, offset=-i) # embed 1's in matrix with correct offset
            head_bsz = head.repeat(bsz,1,1) # replicate for each batch
            head_bsz = head_bsz.reshape(1, head_bsz.shape[0], head_bsz.shape[1], head_bsz.shape[2]) # make 4d
            att = torch.cat((att, head_bsz), 0) # concat so offest matrices become offset tensor
        att = att.permute(1,0, 2,3) # permute so correct shape
         
        return att
    

    def forward(self, X):
        # wanted to include the forward function
        bsz, T, dim = X.size() # batch size, sequence length, input embedding dim
        V = self.get_queries_keys_values(X)
        att = self.compute_causal_attn_matrices(T, bsz) # bsz x nhead x T x T
        
        # get attention
        Z = torch.matmul(att,V)
        
        # transpose to concat heads on head-dim
        Z = Z.permute(0, 2, 1, 3)

        # concatinate heads
        Z = Z.contiguous().view(bsz, T, -1)

        # apply Wo transformation
        Z = self.out_lin_stacked_I.compute(Z)
        return self.out_dropout(Z)
    

class Wo_mat():
    def __init__(self, T, bsz, kW):
        # create stacked Identity matrix 
        # number of I's = nhead = kW
        self.wo = torch.zeros(T*kW, T)
        self.wo = self.wo.fill_diagonal_(1, wrap=True)
        self.wo = self.wo.repeat(bsz, 1, 1)

    def compute(self, X):
        conv = torch.matmul(X, self.wo)
        return conv


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hid_lin = nn.Linear(config.mod_dim, 4 * config.mod_dim, bias=config.bias)
        self.out_lin = nn.Linear(4 * config.mod_dim, config.mod_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        return self.dropout(self.out_lin(F.gelu(self.hid_lin(X))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.mod_dim)
        self.norm2 = nn.LayerNorm(config.mod_dim)
        self.mlp = MLP(config)

    def forward(self, X):
        """
        X - bsz x T x dim
        returns:
           a bsz x T x dim tensor
        """
        Z = X + self.attn(self.norm1(X))
        return Z + self.mlp(self.norm2(Z))


class ConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.mod_dim)
        self.norm2 = nn.LayerNorm(config.mod_dim)
        self.mlp = MLP(config)
        self.conv = nn.Conv1d(config.mod_dim, config.mod_dim, config.kW, padding=config.kW - 1)
        self.pad = config.kW - 1
    

    def forward(self, X):
        """
        X - bsz x T x dim
        returns:
           a bsz x T x dim tensor
        """
        bsz, T, dim = X.size()
        
        # layer norm on X
        X = self.norm1(X)

        # transpose convolve and transpose back
        X_perm = X.permute(0,2,1)
        Z = self.conv(X_perm)
        Z = Z.permute(0,2,1)

        # remove last 4 columns
        Z = Z[:, :-self.pad, :]

        # apply relu
        Z = nn.functional.relu(Z)
        return Z + self.mlp(self.norm2(Z))


class DegenerateBlock(nn.Module):
    """
    a degenerate version of the transformer and conv blocks above,
    consisting just of an MLP and a residual connection.
    """
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.mod_dim)
        self.mlp = MLP(config)

    def forward(self, X):
        """
        X - bsz x T x dim
        returns:
           a bsz x T x dim tensor
        """
        return X + self.mlp(self.norm1(X))


@dataclass
class LMConfig:
    context_length: int = 64
    vocab_size: int = 256
    nlayer: int = 3
    nhead: int = 2
    mod_dim: int = 64
    dropout: float = 0.1
    bias: bool = False
    kW: int = 5


class Decoder(nn.Module):
    def __init__(self, config, model_type="transformer"):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.model_type = model_type
        self.config = config

        self.tok_lookup = nn.Embedding(config.vocab_size, config.mod_dim)
        if model_type == "transformer":
            self.pos_lookup = nn.Embedding(config.context_length, config.mod_dim)

        block_ctor = None
        if model_type == "transformer":
            block_ctor = TransformerBlock
        elif model_type == "cnn":
            block_ctor = ConvBlock
        else:
            block_ctor = DegenerateBlock

        self.blocks = nn.ModuleList([block_ctor(config) for _ in range(config.nlayer)])
        self.drop = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.mod_dim)

        self.lm_head = nn.Linear(config.mod_dim, config.vocab_size, bias=False)
        # tie input and output embeddings
        self.tok_lookup.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('out_lin.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.nlayer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        bsz, T = input_ids.size()
        assert T <= self.config.context_length
   
        X = self.tok_lookup(input_ids) # bsz x T x mod_dim token embeddings

        if self.model_type == "transformer": # add 1 x T x mod_dim positional embeddings
            X = X + self.pos_lookup.weight[:T].unsqueeze(0)

        X = self.drop(X)
        for block in self.blocks:
            X = block(X)
        X = self.norm(X)
        logits = self.lm_head(X)
        return logits

    def configure_optimizer(self, lr, awd):
        no_decay = ["bias", "norm"]
        beta1, beta2 = (0.9, 0.999)
        eps = 1e-06

        # don't decay biases or any LayerNorm params
        grouped_parameters = [
            {"params": [p for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)], "weight_decay": awd,},
            {"params": [p for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},]

        optim = torch.optim.AdamW(grouped_parameters, lr=lr, betas=(beta1, beta2), eps=eps)
        return optim
