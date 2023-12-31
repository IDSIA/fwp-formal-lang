import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_weight import delta_net_fn
from rec_update_fwm_tanh import rec_delta_fn
from fast_transformers import fast_weight_sum
from self_ref_v0 import self_ref_v0


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        seq_len = x.size(0)
        assert seq_len < self.max_len, (
            "Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


# Quasi RNN-like https://arxiv.org/abs/1611.01576
# But the output gate is conditioned by c(t) instead of c(t-1)
class QuasiLSTMlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, forget_bias=0.):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # weight matrices
        self.wm_z = nn.Parameter(torch.rand(hidden_dim, input_dim))
        self.wm_f = nn.Parameter(torch.rand(hidden_dim, input_dim))

        # weight vectors
        self.wv_z = nn.Parameter(torch.rand(1, hidden_dim))  # append B dim
        self.wv_f = nn.Parameter(torch.rand(1, hidden_dim))

        # biases
        self.bias_z = nn.Parameter(torch.rand(1, hidden_dim))
        self.bias_f = nn.Parameter(torch.rand(1, hidden_dim))
        self.forget_bias = forget_bias

        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.wm_z, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.wm_f, mean=0.0, std=0.1)

        torch.nn.init.normal_(self.wv_z, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.wv_f, mean=0.0, std=0.1)

        torch.nn.init.normal_(self.bias_z, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.bias_f, mean=0.0, std=0.1)
        with torch.no_grad():
            self.bias_f.copy_(self.bias_f + self.forget_bias)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        # state is a tuple
        slen, bsz, x_dim = x.size()

        if state is None:
            hidden_prev = torch.zeros(
                [bsz, self.hidden_dim], device=x.device)
        else:
            hidden_prev = state.squeeze(0)  # layer dim compat.

        weight_matrix = torch.cat([self.wm_z, self.wm_f], dim=0)
        out = x.reshape(slen * bsz, x_dim)
        out = F.linear(out, weight_matrix)
        out = out.view(slen, bsz, self.hidden_dim * 2)

        out_z, out_f = torch.split(out, (self.hidden_dim,) * 2, -1)

        output_list = []

        new_cell = hidden_prev
        for z_, f_ in zip(out_z, out_f):
            z_part = torch.tanh(
                z_ + self.wv_z * new_cell + self.bias_z)
            f_part = torch.sigmoid(
                f_ + self.wv_f * new_cell + self.bias_f)
            new_cell = new_cell * f_part + (1. - f_part) * z_part
            output_list.append(new_cell.clone())

        new_cells = torch.stack(output_list)  # (len, B, dim)

        # all outputs
        return new_cells


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, ff_dim, res_dim, dropout, use_layernorm=True,
                 shared_linear1=None, shared_linear2=None, shared_ln=None):
        super(TransformerFFlayers, self).__init__()

        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        if shared_linear1 is not None:
            self.linear1 = shared_linear1
        else:
            self.linear1 = nn.Linear(res_dim, ff_dim)

        if shared_linear2 is not None:
            self.linear2 = shared_linear2
        else:
            self.linear2 = nn.Linear(ff_dim, res_dim)

        self.ff_layers = nn.Sequential(
            self.linear1, nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            self.linear2,
            nn.Dropout(dropout),
        )

        if use_layernorm:
            if shared_ln is not None:
                self.layer_norm = shared_ln
            else:
                self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        out = self.ff_layers(out) + x
        return out


# Fast weight layer with feed-forward fast net
class FastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = delta_net_fn

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        head_q = elu_p1(head_q)
        head_k = elu_p1(head_k)

        # normalize k and q, crucial for stable training.
        head_k = sum_norm(head_k)
        head_q = sum_norm(head_q)

        fast_weights = torch.zeros(
            bsz, self.num_head, self.dim_head, self.dim_head,
            device=head_k.device)

        out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out


# Fast weight layer with feed-forward fast net
class Attentionlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(Attentionlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = fast_weight_sum

        self.slow_net = nn.Linear(
            in_dim, num_head * 3 * dim_head, bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.scale = 1 / (dim_head ** 0.5)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkv = self.slow_net(out)
        qkv = qkv.view(slen, bsz, self.num_head, 3 * self.dim_head)
        head_q, head_k, head_v = torch.split(qkv, (self.dim_head,) * 3, -1)

        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)

        attn_mask = torch.triu(
            head_q.new_ones(slen, slen), diagonal=1).bool()[:, :, None]

        attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        attn_prob = F.softmax(attn_score, dim=1)

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        out = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.num_head * self.dim_head)

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out


# Fast weight layer with feed-forward fast net
class LinearAttentionlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(LinearAttentionlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = fast_weight_sum

        self.slow_net = nn.Linear(
            in_dim, num_head * 3 * dim_head, bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkv = self.slow_net(out)
        qkv = qkv.view(slen, bsz, self.num_head, 3 * self.dim_head)
        head_q, head_k, head_v = torch.split(qkv, (self.dim_head,) * 3, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        head_q = elu_p1(head_q)
        head_k = elu_p1(head_k)

        # # normalize k and q, crucial for stable training.
        # head_k = sum_norm(head_k)
        # head_q = sum_norm(head_q)

        fast_weights = torch.zeros(
            bsz, self.num_head, self.dim_head, self.dim_head,
            device=head_k.device)

        out = self.fw_layer(head_q, head_k, head_v, fast_weights)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out


class SRWMlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout, use_ln=True,
                 use_input_softmax=False, beta_init=-1.0, use_res=True):
        super(SRWMlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.use_ln = use_ln
        self.use_res = use_res
        self.use_input_softmax = use_input_softmax

        self.sr_layer = self_ref_v0
        n_head = num_head
        d_head = dim_head

        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4),
                                requires_grad=True)
        if use_ln:
            self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_parameters(beta_init)

    def reset_parameters(self, beta_init):
        std = 1.0 / math.sqrt(self.dim_head)
        std_q = 0.01 / math.sqrt(self.dim_head)  # new q init
        nn.init.normal_(self.W_y, mean=0., std=std)
        nn.init.normal_(self.W_q, mean=0., std=std_q)
        nn.init.normal_(self.W_k, mean=0., std=std)
        # tried -1 for beta but 0 seems to be better
        # nn.init.normal_(self.w_b, mean=-5., std=std)
        nn.init.normal_(self.w_b, mean=beta_init, std=std)

    def forward(self, h):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = h.size()

        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        if self.use_input_softmax:
            x = F.softmax(x, dim=-1)
        # reshape to (B, heads, len, dim)
        x = x.permute(1, 2, 0, 3)

        W_y_bc = self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = self.w_b.repeat(bsz, 1, 1, 1)

        out = self.sr_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        if self.use_ln:
            if self.use_res:
                out = self.layer_norm(h) + out
        else:
            if self.use_res:
                out = h + out

        return out


# Fast weight layer with feed-forward fast net,
# with recurrent update rule.
class RecUpdateTanhFastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(RecUpdateTanhFastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = rec_delta_fn

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.R_q = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.R_k = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.R_v = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.r_b = nn.Parameter(torch.Tensor(1, num_head, 1, dim_head),
                                requires_grad=True)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.R_q, mean=0., std=std)
        nn.init.normal_(self.R_k, mean=0., std=std)
        nn.init.normal_(self.R_v, mean=0., std=std)
        nn.init.normal_(self.r_b, mean=0., std=std)

    def forward(self, x):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        fast_weights = torch.zeros(
            bsz, self.num_head, self.dim_head, self.dim_head,
            device=head_k.device)

        state0 = torch.zeros(
            bsz, self.num_head, 1, self.dim_head, device=head_k.device)

        out = self.fw_layer(head_q, head_k, head_v, head_beta,
                            self.R_q, self.R_k, self.R_v, self.r_b,
                            fast_weights, state0)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out


if __name__ == '__main__':
    from datetime import datetime
    import random

    torch.manual_seed(123)
    random.seed(123)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    in_vocab_size = 10
    out_vocab_size = 12

    emb_dim = 11
    hidden_size = 8
    dropout = 0.1
    nheads = 2
    dim_head = 4
    num_layers = 2
    ff_factor = 2
    use_pos_enc = False

    # 3, 11, 8
    random_input = torch.tensor(
        [[[5.6525e-01, 1.8138e-01, 1.9374e-01, 2.9642e-01, 8.5111e-01,
          5.4553e-02, 8.8304e-04, 5.2644e-01],
         [4.8553e-01, 7.2578e-01, 5.0821e-01, 4.3623e-01, 6.9831e-01,
          2.1097e-01, 1.6505e-02, 5.6483e-01],
         [7.1936e-01, 4.5909e-01, 7.0579e-01, 7.3716e-01, 3.3913e-01,
          8.0576e-01, 4.8984e-01, 2.8072e-01],
         [8.6652e-01, 6.7032e-01, 1.1779e-01, 1.6276e-02, 3.8614e-01,
          2.6699e-01, 9.2670e-01, 3.3399e-01],
         [1.8828e-01, 1.3678e-01, 9.2607e-01, 2.5674e-01, 2.0871e-01,
          7.0338e-01, 9.1746e-01, 4.9996e-01],
         [1.8541e-01, 3.7135e-01, 7.5193e-01, 2.0723e-01, 6.1000e-01,
          3.2649e-01, 2.9587e-01, 2.0313e-01],
         [8.1343e-01, 9.0418e-01, 2.8495e-01, 3.5581e-02, 9.5951e-01,
          2.0065e-01, 3.5305e-01, 7.0725e-01],
         [9.8584e-01, 4.4119e-01, 6.8732e-01, 2.0627e-01, 7.8432e-01,
          8.3444e-01, 7.5306e-01, 8.0822e-01],
         [5.4248e-01, 6.7067e-01, 1.5366e-01, 4.9530e-01, 9.5891e-01,
          8.5052e-01, 6.7874e-01, 6.9465e-01],
         [2.8098e-01, 2.2131e-01, 6.8086e-01, 1.9270e-01, 3.6595e-01,
          7.7046e-01, 1.5333e-01, 9.6943e-01],
         [2.3490e-01, 1.0462e-01, 1.4274e-01, 9.1545e-01, 9.0547e-02,
          7.7562e-01, 4.7922e-02, 2.3612e-01]],

        [[5.0675e-01, 6.5637e-01, 3.8713e-01, 7.4863e-01, 1.3147e-01,
          1.9421e-01, 8.8610e-02, 2.5563e-01],
         [6.0525e-01, 7.3566e-01, 5.4424e-01, 7.1991e-02, 6.1357e-01,
          8.2875e-01, 5.5580e-01, 4.8719e-01],
         [7.9011e-01, 2.4183e-01, 6.7097e-01, 3.4421e-01, 4.1030e-01,
          7.3783e-01, 1.9415e-01, 5.3634e-01],
         [7.7342e-01, 6.7126e-01, 4.3605e-01, 8.5183e-01, 6.3924e-01,
          5.4433e-01, 8.1974e-01, 9.1918e-01],
         [6.2367e-01, 7.9250e-01, 8.6535e-01, 4.0225e-01, 9.6891e-01,
          1.7830e-01, 5.2103e-01, 9.3327e-01],
         [9.1814e-01, 3.8852e-01, 6.7212e-01, 4.5082e-02, 9.5104e-01,
          1.0365e-01, 6.2249e-01, 2.5153e-01],
         [5.4532e-01, 4.8137e-01, 6.2341e-01, 7.2386e-01, 5.1306e-01,
          8.3670e-01, 7.1230e-01, 5.5039e-01],
         [6.3803e-01, 4.3888e-01, 4.7835e-01, 3.9827e-01, 8.5078e-01,
          3.9567e-01, 7.0928e-01, 4.4533e-01],
         [1.6199e-01, 2.0816e-01, 1.6808e-01, 8.7981e-01, 9.9038e-01,
          8.1762e-01, 6.8890e-01, 2.8055e-01],
         [5.0575e-01, 1.0572e-01, 2.5296e-01, 5.6565e-01, 1.4175e-01,
          1.7075e-01, 1.8281e-02, 4.4822e-01],
         [9.9737e-01, 8.7847e-01, 6.5789e-01, 4.2716e-01, 2.4522e-01,
          5.4319e-01, 7.9834e-01, 3.1782e-01]],
        [[2.6780e-02, 1.4195e-01, 8.5952e-01, 7.9798e-01, 3.9532e-01,
          2.2133e-01, 2.5794e-01, 3.3087e-01],
         [3.5366e-02, 4.5443e-01, 3.1993e-01, 3.1088e-01, 8.7296e-01,
          2.9689e-01, 9.3842e-01, 8.6739e-01],
         [1.0904e-01, 8.3057e-01, 8.5701e-01, 3.5092e-01, 3.8028e-01,
          9.1241e-01, 6.2197e-01, 2.8654e-01],
         [6.7365e-01, 6.1427e-01, 1.1535e-01, 3.3471e-02, 8.4725e-01,
          5.8448e-01, 8.0660e-01, 3.7534e-02],
         [1.2781e-01, 7.9801e-01, 2.4476e-01, 1.3913e-01, 9.4704e-01,
          2.0895e-01, 3.8377e-01, 1.8020e-01],
         [9.1115e-01, 3.5167e-01, 7.4405e-01, 1.8466e-01, 8.9381e-01,
          5.5626e-02, 7.2327e-01, 7.1786e-03],
         [9.0512e-01, 1.7797e-01, 2.7001e-01, 5.5632e-01, 5.7149e-01,
          4.2189e-01, 6.5722e-01, 8.9312e-01],
         [3.0342e-02, 4.1236e-01, 1.6890e-01, 4.2497e-01, 4.3859e-01,
          6.8398e-01, 3.9126e-01, 4.2313e-01],
         [6.3774e-02, 7.2739e-01, 1.6902e-01, 8.7458e-01, 6.4171e-01,
          3.6945e-01, 8.7900e-01, 3.6545e-01],
         [5.5585e-01, 1.3082e-01, 4.3736e-01, 9.5007e-01, 6.0109e-01,
          5.1184e-01, 8.8918e-01, 2.1931e-01],
         [4.9009e-01, 1.5641e-01, 2.2840e-01, 5.8171e-01, 1.5900e-01,
          1.8656e-01, 9.7319e-01, 4.1517e-01]]], device='cuda')

    print("========================")
    print(f"  Test PositionalEncoding  {datetime.now()}")
    print("========================")
    model = PositionalEncoding(hidden_size)

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))

    print("========================")
    print(f"  Test TransformerFFlayers  {datetime.now()}")
    print("========================")

    model = TransformerFFlayers(
        ff_factor * hidden_size, hidden_size, dropout=0.0)

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))

    print("========================")
    print(f"  Test TransformerFFlayers  w/ dropout {datetime.now()}")
    print("========================")

    model = TransformerFFlayers(
        ff_factor * hidden_size, hidden_size, dropout=0.2)

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))

    print("========================")
    print(f"  Test TransformerFFlayers No layernorm {datetime.now()}")
    print("========================")

    model = TransformerFFlayers(
        ff_factor * hidden_size, hidden_size, dropout=0.0, use_layernorm=False)

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))

    print("========================")
    print(f"  Test RecUpdateTanhFastFFlayer  {datetime.now()}")
    print("========================")

    model = RecUpdateTanhFastFFlayer(
        nheads, dim_head, hidden_size, dropout=0.0)

    model = model.to('cuda')
    print("Random input ===> ")
    print(random_input)
    print("Output ===> ")
    print(model(random_input))
