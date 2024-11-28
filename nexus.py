import torch
import math
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from transformers import GPT2Tokenizer
device='cuda' if torch.cuda.is_available() else 'cpu'
    
#---------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class NexusConfig:
    block_size: int = 1024  # context size
    vocab_size: int = 50257  # size of the vocab
    n_layer: int = 24  # number of layers
    n_head: int = 16  # number of attention heads
    n_embd: int = 1024  # number of embeddings
    dropout:float=0.2

class CausalSelfAttention(nn.Module):
    def __init__(self, config:NexusConfig):
        super().__init__()
        self.config=config
        assert config.n_embd%config.n_head==0 ,"Shape mismatch in head"
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.n_head=config.n_head
        self.n_embd=config.n_embd
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
            
    def forward(self,x):
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)    
        att=(q@k.transpose(-2,-1)*(1.0/math.sqrt(k.size(-1))))
        att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        att=F.softmax(att,dim=-1)
        y=att@v
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y
            
class MLP(nn.Module):
    def __init__(self,config:NexusConfig):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)  
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Apply dropout after attention
        x = x + self.mlp(self.ln_2(x))   # Apply dropout after MLP
        return x

            
class Nexus(nn.Module):
    def __init__(self):
        super().__init__()
        config=NexusConfig()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
                    wte=nn.Embedding(config.vocab_size,config.n_embd),
                    wpe=nn.Embedding(config.block_size,config.n_embd),
                    h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    ln_f=nn.LayerNorm(config.n_embd)
                ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight=self.lm_head.weight
        self.tokenizer =GPT2Tokenizer.from_pretrained("Nexus_tokenizer")
        self.load_state_dict(torch.load("weights\\weights.pth",map_location=device))
        self.to(device)
        self.eval()
        print("Model Device :",device)
        
    def forward(self,idx,targets=None):
        B,T=idx.size()
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb=self.transformer.wpe(pos)
        tok_emb=self.transformer.wte(idx)
        x=tok_emb+pos_emb
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,self.config.vocab_size),target=targets.view(-1).to(logits.device))
        return logits,loss
    
    
    def generate(self, x, num_samples=1, max_new_tokens=50, print_outs=False):
        assert max_new_tokens<1020 ,"Max New Tokens cannot be greated than 1020"
        x=f"""[USER <NEUTRAL>]{x}[BOT <NEUTRAL>]"""
        with torch.no_grad():
            # Encode input
            x = self.encoder(x=x).to(device)  # Encoding step
            if x.ndim < 2:
                x = x.unsqueeze(0)  # Add batch dimension if missing
            logits, _ = self(x)  
            samples = []  # To store generated sequences
            # Generate for each sample
            for sample in range(num_samples):
                out = x.clone()  # Copy input for each sample
                # Token generation loop
                for _ in range(max_new_tokens):
                    logits, _ = self(out)  # Forward pass
                    logits = logits[:, -1, :]  # Focus on last token logits
                    probs = F.softmax(logits, dim=-1)  # Convert to probabilities
                    topk_probs, topk_indx = torch.topk(probs, 50, dim=-1)  # Top-k sampling
                    ix = torch.multinomial(topk_probs, 1)  # Sample from top-k
                    xcol = torch.gather(topk_indx, -1, ix)  # Map sampled index to vocab
                    # EOS token check
                    if xcol.item() == self.tokenizer.encode('[END]')[0]:
                        break
                    out = torch.cat((out, xcol), dim=1)  # Append generated token
                samples.append(out.squeeze(0))  # Store completed sequence

            # Decode and return results
            if print_outs:
                for i, sample in enumerate(samples):
                    decoded_output = self.decoder(sample)
                    print(f"Sample {i + 1}: {decoded_output}")
                return None
            else:
                outputs = [self.decoder(sample) for sample in samples]  # Decode all samples
                return outputs

    def encoder(self,x):
        return torch.tensor(self.tokenizer.encode(x))
    def decoder(self, x):
        text = ""
        for item in x:
            try:
                # Attempt to decode the token
                t = self.tokenizer.decode(item)
                text += t
            except KeyError as e:
                continue  # Skip the problematic token or handle as necessary

        return text