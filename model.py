from modules import *

# áp dụng rope để biểu diễn vị trí tương đối cho self attention
def apply_rope(q, k, device, dtype):
    batch_size, n_head, seq_len, head_dim = q.shape

    position = torch.arange(0, seq_len).to(device, dtype)
    dim = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    theta = torch.exp(-math.log(10000.0) * dim / head_dim)

    # theta = 10000 ** (torch.arange(0, head_dim, 2) / head_dim).to(device, torch.float32)
    theta = torch.einsum("i,j->ij", position, theta)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    q1, q2 = q[:, :, :, :head_dim // 2], q[:, :, :, head_dim // 2:]
    k1, k2 = k[:, :, :, :head_dim // 2], k[:, :, :, head_dim // 2:]
    q1, q2, k1, k2 = q1.to(device, dtype), q2.to(device, dtype), k1.to(device, dtype), k2.to(device, dtype)

    q = torch.concat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).to(device, dtype)
    k = torch.concat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).to(device, dtype)
    return q, k


def apply_rope_for_kv_cache(q, k, k_or_v_cache, device, dtype):
    batch_size, n_head, seq_len, head_dim = q.shape
    if k_or_v_cache is not None: pos = k_or_v_cache.shape[2]
    else: pos = 0

    pos = torch.tensor([pos]).to(device, dtype)
    dim = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    theta = torch.exp(-math.log(10000.0) * dim / head_dim)
    # theta = 10000 ** (torch.arange(0, head_dim, 2) / head_dim).to(device, torch.float32)

    theta = torch.einsum("i,j->ij", pos, theta)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    q1, q2 = q[:, :, :, :head_dim // 2], q[:, :, :, head_dim // 2:]
    k1, k2 = k[:, :, :, :head_dim // 2], k[:, :, :, head_dim // 2:]
    q1, q2, k1, k2 = q1.to(device, dtype), q2.to(device, dtype), k1.to(device, dtype), k2.to(device, dtype)

    q = torch.concat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).to(device, dtype)
    k = torch.concat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).to(device, dtype)
    return q, k


def scaled_dot_product_attention(q, k, v, mask=None, device=None, dtype=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    if mask is not None:
        attn_logits += mask

    attention = F.softmax(attn_logits.to(dtype=torch.float32), dim=-1).to(device, dtype)
    values = torch.matmul(attention, v)
    return values, attention


def repeat_kv(x, n_repeat):
    if n_repeat == 1: return x
    B, S, H, D = x.shape
    x = x.unsqueeze(3)
    x = x.expand(-1, -1, -1, n_repeat, -1)
    return x.reshape(B, S, H * n_repeat, D)


class SelfAttentionHeads(nn.Module):
    def __init__(self, d_model=512, kv_d_model=128, n_q_head=8, n_kv_head=2, n_ffn=2048, device=None, dtype=torch.float32):
        super(SelfAttentionHeads, self).__init__()
        self.d_model = d_model
        self.n_ffn = n_ffn
        self.device = device
        self.dtype = dtype

        # các tham số cho GQA (Group Query Attention)
        self.kv_d_model = kv_d_model
        self.q_head_dim = d_model // n_q_head
        self.kv_head_dim = kv_d_model // n_kv_head
        self.n_q_head = n_q_head
        self.n_kv_head = n_kv_head

        # các lớp tuyến tính chiếu cho trọng số attention
        self.qkv_linear = nn.Linear(d_model, d_model + (kv_d_model * 2), bias=True, device=device, dtype=dtype)
        self.linear_projection = nn.Linear(d_model, d_model, bias=True, device=device, dtype=dtype) # lớp chiếu attention values

        # cache
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, mask=None, kv_cache=False, limit_context=1024):
        x = x.to(self.device, self.dtype)
        batch_size, seq_len, d_model = x.shape
        qkv_out = self.qkv_linear(x)

        # GQA (Group Query Attention)
        q, k, v = (
            qkv_out[:, :, :self.d_model],
            qkv_out[:, :, self.d_model:self.d_model + self.kv_d_model],
            qkv_out[:, :, self.d_model + self.kv_d_model:]
        )

        q = q.view(batch_size, seq_len, self.n_q_head, self.q_head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_head, self.kv_head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_head, self.kv_head_dim)

        # điều chỉnh chiều KV phù hợp với chiều của Q qua repeat
        k = repeat_kv(k, int(self.n_q_head / self.n_kv_head))
        v = repeat_kv(v, int(self.n_q_head / self.n_kv_head))

        # transpose từ (B, S, H, D) về (B, H, S, D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # áp dụng kv_cache cho rope
        if kv_cache: q, k = apply_rope_for_kv_cache(q, k, self.k_cache, self.device, self.dtype)
        else: q, k = apply_rope(q, k, self.device, self.dtype)

        # nối lại k, v với token mới và lưu lại
        if self.k_cache is not None and kv_cache:
            self.k_cache = torch.concat([self.k_cache, k], dim=2)
            self.v_cache = torch.concat([self.v_cache, v], dim=2)
        if self.k_cache is None and kv_cache:
            self.k_cache = k
            self.v_cache = v

        # kiểm tra limit context trong kv_cache và cắt context
        if self.k_cache is not None and self.k_cache.shape[2] >= limit_context and kv_cache:
            self.k_cache = self.k_cache[:, :, -limit_context:, :]
            self.v_cache = self.v_cache[:, :, -limit_context:, :]

        # tính attention
        if kv_cache:
            values, attention = scaled_dot_product_attention(q, self.k_cache, self.v_cache, mask, self.device, self.dtype)
        else:
            values, attention = scaled_dot_product_attention(q, k, v, mask, self.device, self.dtype)
        values = values.transpose(1, 2)
        values = values.reshape(batch_size, seq_len, self.n_q_head * self.q_head_dim)

        out = self.linear_projection(values)
        return out
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model=512, n_ffn=2048, device=None, dtype=None):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, n_ffn, bias=True, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        self.ffn_dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(n_ffn, d_model, bias=True, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        x = x.to(self.device, self.dtype)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.ffn_dropout(x)
        x = self.linear2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, kv_d_model=128, n_q_head=8, n_kv_head=2, n_ffn=2048, device=None, dtype=None):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.kv_d_model = kv_d_model
        self.n_q_head = n_q_head
        self.n_kv_head = n_kv_head
        self.n_ffn = n_ffn
        self.device = device
        self.dtype = dtype

        self.mha = SelfAttentionHeads(d_model, kv_d_model, n_q_head, n_kv_head, n_ffn, device, dtype)
        self.ffn = FeedForwardBlock(d_model, n_ffn, device, dtype)

        self.layernorm1 = nn.LayerNorm(d_model, device=device, dtype=torch.float32) # dùng float 32 cho layernorm đảm bảo chính xác số học
        self.layernorm2 = nn.LayerNorm(d_model, device=device, dtype=torch.float32)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask=None, kv_cache=False, limit_context=1024):
        x = x.to(self.device, self.dtype)

        x_prenorm = self.layernorm1(x.to(dtype=torch.float32)).to(self.device, self.dtype) # pre norm
        attn_out = self.mha(x_prenorm, mask, kv_cache, limit_context)
        x = x + self.dropout1(attn_out)

        x_prenorm = self.layernorm2(x.to(dtype=torch.float32)).to(self.device, self.dtype) # pre norm
        ffn_out = self.ffn(x_prenorm)
        x = x + self.dropout2(ffn_out)

        return x
    
class Causal(nn.Module):
     def __init__(self, vocab_size, d_model=512, kv_d_model=128, n_q_head=8, n_kv_head=2, n_ffn=2048, n_layer=2, padding_idx=-1, device=None, dtype=None):
         super(Causal, self).__init__()
         self.vocab_size = vocab_size
         self.d_model = d_model
         self.n_q_head = n_q_head
         self.n_kv_head = n_kv_head
         self.n_ffn = n_ffn
         self.n_layer = n_layer
         self.device = device
         self.dtype = dtype

         # layers
         self.layers = nn.ModuleList([
              TransformerBlock(d_model, kv_d_model, n_q_head, n_kv_head, n_ffn, device, dtype)
              for _ in range(n_layer)
          ])

         # weight tying embed/out
         self.embed = nn.Embedding(vocab_size, d_model, padding_idx, device=device, dtype=dtype)
         nn.init.xavier_uniform_(self.embed.weight)
         self.linear_out = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
         self.linear_out.weight = self.embed.weight

     def create_mask(self, x):
         batch_size, seq_len = x.shape
         causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
         mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))
         return mask

     def reset_kv_cache(self):
        for layer in self.layers:
            layer.mha.k_cache = None
            layer.mha.v_cache = None
        return layer.mha.v_cache

     def get_kv_cache(self):
         kv_cache_shape, kv_cache = list(), list()
         for i in range(len(self.layers)):
             if self.layers[i].mha.k_cache is None: return None
             kv_cache_shape.append((f"Layer {i+1}", self.layers[i].mha.k_cache.shape, self.layers[i].mha.v_cache.shape))
             kv_cache.append((self.layers[i].mha.k_cache, self.layers[i].mha.v_cache))
         return kv_cache, kv_cache_shape

     def load_kv_cache(self, kv_cache):
        for i in range(len(self.layers)):
            self.layers[i].mha.k_cache = kv_cache[i][0]
            self.layers[i].mha.v_cache = kv_cache[i][1]
        return self.layers[i].mha.v_cache.shape

     def update_kv_cache(self, tokenizer, input):
         token = tokenizer.batch_encode(input, return_tensor_type="tensor")["input_ids"].to(device)
         output = self.generate(token, limit_context=1024, max_token=token.shape[-1], top_k=1)
         return self.layers[-1].mha.v_cache.shape

     def forward(self, x, padding_mask=None, kv_cache=False, limit_context=1024):
         x = x.to(device)

         # khởi tạo mask
         if padding_mask is not None:
              if kv_cache:
                  raise ValueError("kv_cache không được hỗ trợ với padding_mask")
              padding_mask = padding_mask == 0
              padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf'))
              mask = padding_mask.unsqueeze(1).unsqueeze(1) + self.create_mask(x)
              mask = mask.to(self.device, self.dtype)
         elif kv_cache:
              # tắt mask khi kv cache
              mask = None
         else:
              mask = self.create_mask(x).unsqueeze(0).unsqueeze(0)
              mask = mask.to(self.device, self.dtype)

         # lớp nhúng
         x = self.embed(x)

         # các lớp chú ý, truyền thẳng
         for layer in self.layers:
              x = layer(x, mask, kv_cache, limit_context)

         return self.linear_out(x)

     def get_model_loss(self, test_input, tokenizer):
         self.eval()
         with torch.no_grad():
             criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
             x = tokenizer.batch_encode(test_input, return_tensor_type="tensor")
             x, mask = x['input_ids'].to(device), x['attention_mask'].to(device)
             x, mask = x[:, :256], mask[:, :256]
             xb = x.clone()[:, :-1].to(device)
             yb = x.clone()[:, 1:].to(device)
             mask = mask[:, :-1].to(device)
             output = self(xb, mask)
             loss = criterion(output.reshape(-1, self.vocab_size), yb.reshape(-1))
             del(criterion)
             return loss.item()

     def generate(self, inputs, limit_context=1024, max_token=20, top_k=4, top_p=0.9, temperature=0.7, repeat_penalty=1.2, print_per_token=False, tokenizer=None):
         with torch.no_grad():

             hist_tokens = list()
             # các thuật toán lấy mẫu
             def sampling(probs, top_k=3, top_p=0.8, temperature=0.7):
                 probs /= temperature
                 # repeat penalty
                 for token in hist_tokens:
                     probs[:, token] /= repeat_penalty
                 probs = torch.softmax(probs, dim=-1)

                 # switch to greedy search if topk is 1
                 if top_k == 1:
                     token = torch.argmax(probs, dim=-1).unsqueeze(0)
                     if int(token[0]) not in hist_tokens:
                         hist_tokens.append(int(token[0]))
                     return token

                 # top_k
                 top_k = min(top_k, probs.shape[-1])
                 values, indices = torch.topk(probs, k=top_k, dim=-1)
                 values = values.flatten()
                 indices = indices.flatten()

                 # top_p
                 for cut in range(values.shape[-1]):
                     if values[:cut].sum() >= top_p: break
                 values, indices = values[:cut], indices[:cut]
                 idx = torch.multinomial(values, num_samples=1)

                 # save indices history
                 if int(indices[idx][0]) not in hist_tokens:
                     hist_tokens.append(int(indices[idx][0]))

                 return indices[idx].unsqueeze(0)

             def generate_(model, inputs, max_token=20, top_k=4, top_p=0.9, temperature=0.7, print_per_token=False, tokenizer=None):
                 if print_per_token and tokenizer is None:
                     raise ValueError("'print per token' option need tokenizer!")

                 for i in range(max_token):
                     output = model(inputs[:, i:i+1], kv_cache=True, limit_context=limit_context)

                     if i >= (inputs.shape[1] - 1):
                         next_token = sampling(output[:, 0, :], top_k, top_p, temperature)
                         inputs = torch.concat([inputs, next_token], dim=-1)
                         if print_per_token and tokenizer is not None:
                             print(tokenizer.decode(next_token[0]), end="", flush=True)

                 if print_per_token and tokenizer is not None:
                     print()
                 return inputs

             return generate_(self, inputs, max_token, top_k, top_p, temperature, print_per_token, tokenizer)
         
