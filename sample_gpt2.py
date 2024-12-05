'''
Sample from the model trained in train_gpt2.py. Meant to be run on a single GPU/CPU.
'''
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# Import necessary components from train_gpt2.py
from train_gpt2 import GPT, GPTConfig, norm, CastedLinear, create_block_mask

# Prepare the tokenizer using tiktoken as in fineweb.py
import tiktoken

# Set the path to the checkpoint
checkpoint_path = '/Users/davisbrown/logs/ceef0f8f-86c5-493a-ab66-2f2b4e5795a3/state_step001750.pt'

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model configuration
num_vocab = 50304  # Extended to nearest multiple of 128 for efficiency
config = GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768)

# Initialize the model
model = GPT(config)
model = model.to(device).bfloat16()
# Cast CastedLinear layers to float as in training
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()

# Load the checkpoint and fix the state dict keys
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model']

# Remove '_orig_mod.' prefix from state dict keys
fixed_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('_orig_mod.'):
        fixed_state_dict[k[10:]] = v  # Remove '_orig_mod.' (10 characters)
    else:
        fixed_state_dict[k] = v

# Load the fixed state dict
model.load_state_dict(fixed_state_dict)

# Set the model to evaluation mode
model.eval()

# Initialize the tokenizer using tiktoken
enc = tiktoken.get_encoding('gpt2')
eot_token = enc.eot_token  # End-of-text token ID

# Modify the GPT class to include a generate_forward method
def generate_forward(self, idx, v1, attn_blocksize):
    """Generate logits for the next token."""
    S = idx.size(1)
    docs = (idx == eot_token).cumsum(1)
    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = docs[0, q_idx] == docs[0, kv_idx]
        window_mask = q_idx - kv_idx < attn_blocksize
        return causal_mask & document_mask & window_mask
    block_mask = create_block_mask(
        document_causal_mask, None, None, S, S, device=idx.device, _compile=True
    )

    # Forward pass
    x = self.transformer.wte(idx)  # Token embeddings
    x = norm(x)
    x0 = x

    # Store outputs for U-Net skip connections
    skip_connections = []

    # Encoder pass
    for i in range(self.num_encoder_layers):
        x, v1 = self.transformer.h[i](x, v1, x0, block_mask)
        skip_connections.append(x)

    # Decoder pass with weighted skip connections
    for i in range(self.num_decoder_layers):
        x = x + self.skip_weights[i] * skip_connections.pop()
        x, v1 = self.transformer.h[self.num_encoder_layers + i](x, v1, x0, block_mask)

    x = norm(x)
    logits = self.lm_head(x)
    logits = 30 * torch.tanh(logits / 30)  # Logit clamping as in training
    return logits, v1

# Monkey-patch the generate_forward method to the GPT class
GPT.generate_forward = generate_forward

def generate(model, enc, context, max_new_tokens, temperature=1.0, top_k=None):
    # Encode the context using tiktoken
    context_tokens = [eot_token]  # Start with EOT token as in fineweb.py
    context_tokens.extend(enc.encode(context))
    idx = np.array(context_tokens, dtype=np.uint16)
    idx = torch.tensor(idx, device=device).unsqueeze(0)

    generated = idx
    v1 = None  # Initialize key/value cache
    attn_blocksize = torch.tensor(1792, device=device)  # Max attention block size

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, v1 = model.generate_forward(generated, v1, attn_blocksize)

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
    context = "Once upon a time"
    output_text = generate(
        model, enc, context, max_new_tokens=100, temperature=1.0, top_k=50
    )
    print(output_text) 