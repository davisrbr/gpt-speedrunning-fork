'''
Sample from the model trained in train_gpt2.py. Meant to be run on a single GPU/CPU.
'''
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import tiktoken

# Import FlexAttention dependencies
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# Set the path to the checkpoint
checkpoint_path = 'state_step001750.pt'

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the norm function
def norm(x):
    return F.rms_norm(x, (x.size(-1),))

# Define CastedLinear
class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

# Define Rotary
class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # Apply rotary embeddings
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

# Define CausalSelfAttention with modifications
class CausalSelfAttention(nn.Module):

    def __init__(self, dim, n_head):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        # Value residual lambda
        self.lamb = nn.Parameter(torch.tensor(0.5))  # @Grad62304977
        # Rotary embeddings
        self.rotary = Rotary(dim // n_head)  # dim // n_head = head_dim
        # Output projection
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_()  # Zero init suggested by @Grad62304977

    def forward(self, x, v1, block_mask):
        B, T = x.size(0), x.size(1)  # Batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)

        if v1 is None:
            v1 = v.clone()  # Initialize v1
        else:
            # Ensure v1 has the same sequence length as v
            v1_seq_len = v1.size(1)
            if v1_seq_len < T:
                # Pad v1 to match the current sequence length
                pad_size = T - v1_seq_len
                pad = torch.zeros(B, pad_size, self.n_head, v.size(-1),
                                  device=v.device, dtype=v.dtype)
                v1 = torch.cat([v1, pad], dim=1)
            elif v1_seq_len > T:
                # Trim v1 to match the current sequence length
                v1 = v1[:, -T:, :, :]

        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)  # @Grad62304977
        q, k = norm(q), norm(k)  # QK norm suggested by @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2),
                           v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x)  # Re-assemble all head outputs
        y = self.c_proj(y)
        return y, v1

# Define MLP
class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()  # Zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # Activation function
        x = self.c_proj(x)
        return x

# Define Block
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.n_embd, config.n_head)
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(norm(x), v1, block_mask)
        x = x + x1
        x = x + self.mlp(norm(x))
        return x, v1

# Initialize the tokenizer using tiktoken
enc = tiktoken.get_encoding('gpt2')
eot_token = enc.eot_token  # End-of-text token ID

# Initialize the model configuration
num_vocab = 50304  # Extended to nearest multiple of 128 for efficiency
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = num_vocab
    n_layer: int = 12
    n_head: int = 6  # head_dim 128 suggested by @Grad62304977
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()  # @Grad62304977

    def generate_forward(self, idx, v1_list, attn_blocksize):
        B, T = idx.size(0), idx.size(1)

        # Create the causal mask
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            None,
            None,
            T,
            T,
            device=idx.device,
            _compile=False
        )

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x

        skip_connections = []
        for i in range(self.num_encoder_layers):
            v1 = v1_list[i]
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
            v1_list[i] = v1  # Update v1_list for this layer
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            v1 = v1_list[self.num_encoder_layers + i]
            x = x + self.skip_weights[i] * skip_connections.pop()
            x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)
            v1_list[self.num_encoder_layers + i] = v1  # Update v1_list

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)

        return logits, v1_list

def generate(model, enc, context, max_new_tokens, temperature=1.0, top_k=None):
    # Encode the context using tiktoken
    context_tokens = [eot_token]
    context_tokens.extend(enc.encode(context))
    idx = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
    v1_list = [None] * len(model.transformer.h)  # Initialize v1_list for each layer
    generated = idx
    attn_blocksize = torch.tensor(1792, device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, v1_list = model.generate_forward(generated, v1_list, attn_blocksize)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            topk_values, _ = torch.topk(logits, top_k)
            logits[logits < topk_values[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    output_tokens = generated[0].tolist()
    output_text = enc.decode(output_tokens)
    return output_text

if __name__ == '__main__':

    model = GPT(GPTConfig())
    model = model.to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()

    # load the checkpoint and fix the state dict keys
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    # remove '_orig_mod.' prefix from state dict keys
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k[10:]] = v  # Remove '_orig_mod.'
        else:
            fixed_state_dict[k] = v

    model.load_state_dict(fixed_state_dict)
    model.eval()

    print("Generate some text (Ctrl+C to exit):")
    while True:
        try:
            context = input("> ")
            output_text = generate(
                model, enc, context, max_new_tokens=100, temperature=0.3, top_k=40
            )
            print("\nGenerated text:")
            print(output_text)
            print("\nGenerate some text (Ctrl+C to exit):")
        except KeyboardInterrupt:
            print("\nExiting...")
            break