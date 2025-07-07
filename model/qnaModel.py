import torch
import torch.nn as nn
import torch.nn.functional as F

# === 하이퍼파라미터 전역 변수 (예시값으로 설정) ===
VOCAB_SIZE = 30522
EMB_DIM = 512
CONTEXT_LENGTH = 128
NUM_HEADS = 8
NUM_LAYERS = 6
DROP_RATE = 0.1
QKV_BIAS = False

# === 레이어 정의 ===

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMB_DIM, 4 * EMB_DIM),
            GELU(),
            nn.Linear(4 * EMB_DIM, EMB_DIM),
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        assert d_out % NUM_HEADS == 0
        self.head_dim = d_out // NUM_HEADS
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(DROP_RATE)
        self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_query(x).view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)

        attn_scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask = self.mask[:T, :T].bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, self.d_out)

        return self.out_proj(self.dropout(attn_output))

class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        assert d_out % NUM_HEADS == 0
        self.head_dim = d_out // NUM_HEADS
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.W_key = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.W_value = nn.Linear(d_in, d_out, bias=QKV_BIAS)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(DROP_RATE)

    def forward(self, x, context):
        B, T, _ = x.shape
        S = context.shape[1]

        Q = self.W_query(x).view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)
        K = self.W_key(context).view(B, S, NUM_HEADS, self.head_dim).transpose(1, 2)
        V = self.W_value(context).view(B, S, NUM_HEADS, self.head_dim).transpose(1, 2)

        attn_scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, self.d_out)

        return self.out_proj(self.dropout(attn_output))

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = MultiHeadAttention(EMB_DIM, EMB_DIM)
        self.ff = FeedForward()
        self.norm1 = LayerNorm(EMB_DIM)
        self.norm2 = LayerNorm(EMB_DIM)
        self.drop = nn.Dropout(DROP_RATE)

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_att = MultiHeadAttention(EMB_DIM, EMB_DIM)
        self.cross_att = CrossAttention(EMB_DIM, EMB_DIM)
        self.ff = FeedForward()
        self.norm1 = LayerNorm(EMB_DIM)
        self.norm2 = LayerNorm(EMB_DIM)
        self.norm3 = LayerNorm(EMB_DIM)
        self.drop = nn.Dropout(DROP_RATE)

    def forward(self, x, encoder_output):
        x = x + self.drop(self.self_att(self.norm1(x)))
        x = x + self.drop(self.cross_att(self.norm2(x), encoder_output))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x

# === 전체 인코더-디코더 모델 ===

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)
        self.drop = nn.Dropout(DROP_RATE)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.norm = LayerNorm(EMB_DIM)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.blocks(x)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)
        self.drop = nn.Dropout(DROP_RATE)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(NUM_LAYERS)])
        self.norm = LayerNorm(EMB_DIM)
        self.output = nn.Linear(EMB_DIM, VOCAB_SIZE)

    def forward(self, x, encoder_out):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, encoder_out)

        x = self.norm(x)
        return self.output(x)

class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        logits = self.decoder(tgt, enc_out)
        return logits
