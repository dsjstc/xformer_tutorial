import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define a self-attention layer with multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        d_k = query.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        linear_transformed = []
        for ll, x in zip(self.linear_layers, (query, key, value)):
            head = ll(x)
            linear_transformed.append(head)
        reshaped_tensors = []
        for lt in linear_transformed:
            perm = lt.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            reshaped_tensors.append(perm)

        # Print tensor shapes for debugging
        for i, tensor in enumerate(reshaped_tensors):
            print(f"Tensor {i + 1} Shape: {tensor.shape}")

        # Use the reshaped tensors in further calculations
        query, key, value = reshaped_tensors
        output, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.output_linear(output), attention_weights

# Define a position-wise feed-forward network
class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, d_ff)
        self.fc2 = nn.Linear(d_ff, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define an encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(input_size, d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(input_size, d_ff, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.multi_head_attention(query=x, key=x, value=x, mask=mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(ffn_output)
        x = self.layer_norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, input_size, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(input_size, d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Define a decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention1 = MultiHeadAttention(input_size, d_model, num_heads, dropout=dropout)
        self.multi_head_attention2 = MultiHeadAttention(input_size, d_model, num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn1, _ = self.multi_head_attention1(query=x, key=x, value=x, mask=tgt_mask)
        x = x + self.dropout(attn1)
        x = self.layer_norm1(x)

        attn2, _ = self.multi_head_attention2(query=x, key=enc_output, value=enc_output, mask=src_mask)
        x = x + self.dropout(attn2)
        x = self.layer_norm2(x)

        ffn_output = self.feed_forward(x)
        x = x + self.dropout(ffn_output)
        x = self.layer_norm3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, input_size, d_model, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(input_size, d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask, enc_output, src_mask):
        x = x + self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

# Define a basic positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros(1, max_seq_length, d_model)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1)].clone().detach()

class SemanticSegmentationTransformer(nn.Module):
    def __init__(self, num_classes, input_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, max_seq_length, dropout=0.1):
        super(SemanticSegmentationTransformer, self).__init__()

        self.encoder = Encoder(num_layers=num_encoder_layers, input_size=input_size, d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
        self.decoder = Decoder(num_layers=num_decoder_layers, input_size=input_size, d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_length=max_seq_length, dropout=dropout)
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(x=src, mask=src_mask)
        dec_output = self.decoder(x=tgt, tgt_mask=tgt_mask, enc_output=enc_output, src_mask=src_mask)
        output = self.fc(dec_output)
        return output
