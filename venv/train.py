import torch
import torch.nn as nn
import torch.nn.functional as F

text = "To be, or not to be, that is the question. To build an LLM, that is the answer."


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encoder(text), dtype=torch.long)


batch_size = 4
block_size = 8
n_embd = 32

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch()


class CasualAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 

        v = self.value(x)
        out = wei @ v 
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
    
class miniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = CasualAttentionHead(n_embd)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb 
        x = x + self.sa_head(x) 
        x = x + self.ffwd(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = miniGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

logits, loss = model(xb, yb)

print("ARCH COMPLETED")
print(f"Logits shape (predictions): {logits.shape}")
print(f"Initial Loss value: {loss.item():.4f}\n")

print ("untrained model output:")
context = torch.zeros((1,1), dtype=torch.long)
print(decoder(model.generate(context, max_new_tokens=100)[0].tolist()))
print("-" * 40)

print ("\nTraining the model...")
for step in range(1000):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}: loss {loss.item():.4f}")   

print ("-" * 40)
print ("trained model output:")
print(decoder(model.generate(context, max_new_tokens=100)[0].tolist()))