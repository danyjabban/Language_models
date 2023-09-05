import sys
import math
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import language_models as models

train_fi = "wikitext2_raw_noat_train.txt"
val_fi = "wikitext2_raw_noat_validation.txt"

def get_data(fi, bsz, max_bytes=0):
    with open(fi) as f:
        data = str.encode(f.read()) # get data as bytes
    if max_bytes > 0:
        data = data[:max_bytes]
    btensor = torch.ByteTensor([byte for byte in data])
    # ensure total bytes is a multiple of bsz
    total_len = bsz * (btensor.size(0) // bsz) 
    btensor = btensor[:total_len].view(bsz, -1) # bsz x total_len/bsz
    return btensor

def to_text(x):
    return ''.join(chr(b.item()) for b in x)

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, data, context_length):
        """
        data - bsz x total_len
        """
        super().__init__()
        self.data = data
        self.context_length = context_length
        self.nbatches = data.size(1) // context_length + (0 if data.size(1) % context_length == 0 else 1)
        self.total_len = data.size(1)
    
    def __len__(self):
        return self.nbatches
    
    def __getitem__(self, i):
        """
        returns inputs and targets
        """
        last_idx = min((i+1)*self.context_length, self.total_len-1)
        return (self.data[:, i*self.context_length:last_idx], 
                self.data[:, i*self.context_length+1:last_idx+1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["bigram", "cnn", "transformer", "transcnn"])
    parser.add_argument("--bsz", type=int, default=32, help="training batch size")
    parser.add_argument("--context_length", type=int, default=64, help="training sequence length")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1, help="training batch size")
    parser.add_argument("--seed", type=int, default=59006)
    parser.add_argument("--max_train_bytes", type=int, default=5000000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_dataset = LMDataset(
        get_data(train_fi, args.bsz, max_bytes=args.max_train_bytes), args.context_length)
    val_dataset = LMDataset(get_data(val_fi, 16), args.context_length)

    config = models.LMConfig(dropout=args.drop)

    if args.model_type == "transcnn": # just testing equivalence...
        with torch.inference_mode():
            config.bias = False
            config.dropout = 0.0
            config.context_length = 16
            conv = nn.Conv1d(
                config.mod_dim, config.mod_dim, config.kW, bias=False)
            ccsa = models.CausalConvSelfAttention(conv, config)
            X = torch.randn(2, config.context_length, config.mod_dim)
            Z1 = conv(X.transpose(1, 2)).transpose(1, 2)
            Z2 = ccsa(X)
            print("norm of difference:", torch.norm(Z1[:, -10:] - Z2[:, -10:]))
        sys.exit(0)

    model = models.Decoder(config, model_type=args.model_type)

    optimizer = model.configure_optimizer(args.lr, args.wd)

    def eval_lm(ep):
        with torch.inference_mode():
            model.eval()
            token_loss, ntokens = 0.0, 0
            for step in range(len(val_dataset)):
                inputs, tgts = val_dataset[step]
                inputs, tgts = inputs.long(), tgts.long()
                logits = model(inputs)
                loss = models.token_xent_sum_loss(logits, tgts)
                token_loss += loss.item()
                ntokens += tgts.nelement()
            token_loss /= ntokens
        print("val epoch {:d}: token_loss {:.4f} | ppl {:.3f} | bpc {:.3f}".format(
              ep, token_loss, math.exp(token_loss), token_loss/math.log(2)))
    
    for epoch in range(1, args.epochs+1):
        model.train()
        token_loss, ntokens = 0.0, 0
        ep_start = time.perf_counter()
        #for step, (inputs, tgts) in enumerate(train_loader):
        for step in range(len(train_dataset)):
            optimizer.zero_grad()
            inputs, tgts = train_dataset[step]
            inputs, tgts = inputs.long(), tgts.long()
            logits = model(inputs)
            loss = models.token_xent_sum_loss(logits, tgts)
            token_loss += loss.item()
            ntokens += tgts.nelement()
            loss.div(tgts.nelement()).backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            if (step+1) % args.log_interval == 0:
                print("step {:d}/{:d}: token_loss {:.4f}".format(
                    step+1, len(train_dataset), token_loss/ntokens))

        token_loss /= ntokens
        print("train epoch {:d} (took {:.1f}s): token_loss {:.4f} | ppl {:.3f} | bpc {:.3f}".format(
            epoch, time.perf_counter() - ep_start, token_loss, math.exp(token_loss), token_loss/math.log(2)))
        
        eval_lm(epoch)
