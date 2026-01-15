# Code reused from https://github.com/arghosh/AKT.git
import os

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
import random
from config import Config

# ===== Added for code sequence =====
from codeEncoder import CodeSequenceTransformerEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

# ===== MoE fusion =====
class MoEFusion(nn.Module):
    """
    Multi-Expert Fusion for PKT + code
    Expert1: d_output (题目/交互特征)
    Expert2: code_h (代码特征)
    Expert3: combined residual
    """
    def __init__(self, input_dim, output_dim, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        assert num_experts == 3, "MoEFusion currently supports 3 experts only."

        # Experts
        self.expert_q = nn.Sequential(
            nn.Linear(input_dim // 2, output_dim),
            nn.ReLU()
        )
        self.expert_c = nn.Sequential(
            nn.Linear(input_dim - input_dim // 2, output_dim),
            nn.ReLU()
        )
        self.expert_f = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x: (B, T, input_dim) = torch.cat([d_output, code_h], dim=-1)
        """
        gate_logits = self.gate(x)            # (B, T, 3)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        d_dim = x.size(-1) // 2
        e1 = self.expert_q(x[:, :, :d_dim])
        e2 = self.expert_c(x[:, :, d_dim:])
        e3 = self.expert_f(x)

        experts = torch.stack([e1, e2, e3], dim=-1)  # (B, T, output_dim, 3)
        fused = torch.sum(experts * gate_weights.unsqueeze(-2), dim=-1)  # (B, T, output_dim)
        return fused

class DACE(nn.Module):
    def __init__(
            self,
            n_question,
            n_pid,
            d_model,
            n_blocks,
            kq_same,
            dropout,
            model_type,
            final_fc_dim=512,
            n_heads=8,
            d_ff=2048,
            l2=1e-5,
            separate_qa=False,
            # ===== Added for code sequence =====
            code_emb_dim=768,
            code_hidden_dim=128,
    ):
        super().__init__()

        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        self.device = device

        embed_l = d_model

        if self.n_pid > 0:
            self.s_embed = nn.Embedding(self.n_question + 1, embed_l)
            self.p_embed = nn.Embedding(self.n_pid + 1, embed_l)

        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.pa_embed = nn.Embedding(2, embed_l)

        # ===== DACE backbone (unchanged) =====
        self.model = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model // n_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
        )

        # ===== Added for code sequence =====
        self.code_encoder = CodeSequenceTransformerEncoder(
            code_emb_dim=code_emb_dim,
            d_model=d_model,  # ← 对齐 AKT
            n_heads=n_heads,
            n_layers=2,
            d_ff=d_ff,
            dropout=dropout
        )
        # ===== MoE fusion =====
        self.moe_fusion = MoEFusion(
            input_dim=d_model * 2,
            output_dim=d_model,        # 统一回到 d_model 维
            num_experts=3
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)


        # ===== Output layer (only dimension changed) =====
        self.out = nn.Sequential(
            nn.Linear(d_model, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        init(self)

    # ===== CL loss (unchanged) =====
    def get_cl_loss(self, z1, z2, mask):
        cos = nn.CosineSimilarity(dim=-1)
        cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")

        pooled_z1 = (z1 * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        pooled_z2 = (z2 * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)

        sim = cos(pooled_z1.unsqueeze(1), pooled_z2.unsqueeze(0))
        labels = torch.arange(sim.shape[0]).long().to(device)
        cl_loss = cl_loss_fn(sim, labels)
        return cl_loss

    def forward(
            self,
            q_data,
            pa_data,
            target,
            pid_data=None,
            # ===== Added for code sequence =====
            code_emb=None,
            response=None,
            seq_len=None,
            code_len=None,
            return_output=False,
    ):

        # --------------------------------------------------
        # [ADD] response fallback (MUST, before pa_data overwrite)
        # --------------------------------------------------
        if response is None:
            response = (pa_data - q_data) // self.n_question

        # ===== Question / interaction branch (unchanged) =====
        p_embed_data = self.p_embed(pid_data)
        s_embed_data = self.s_embed(q_data)
        p_embed_data = p_embed_data + s_embed_data

        pa_data = (pa_data - q_data) // self.n_question
        pa_embed_data = self.pa_embed(pa_data) + p_embed_data

        d_output = self.model(p_embed_data, pa_embed_data)
        # --------------------------------------------------
        # [ADD] device (MUST)
        # --------------------------------------------------
        device = d_output.device

        # ===== Added for code sequence =====
        if code_emb is not None:
            code_h = self.code_encoder(
                code_emb=code_emb,
                response=response,
                seq_len=code_len,  # ← 注意：这里明确只用 code_len
            )
        else:
            B, T, _ = d_output.size()
            code_h = torch.zeros(B, T, d_output.size(-1), device=device)
        # 1. 构造 code padding mask
        # --------------------------------------------------
        B, T, _ = code_h.size()
        code_attn_mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) >= code_len.unsqueeze(1)
        # code attends question
        code_ctx, _ = self.cross_attn(
            query=code_h,
            key=d_output,
            value=d_output,
            key_padding_mask=code_attn_mask
        )
        code_h = self.cross_norm(code_h + code_ctx)

        # question attends code
        q_ctx, _ = self.cross_attn(
            query=d_output,
            key=code_h,
            value=code_h,
            key_padding_mask=code_attn_mask
        )
        d_output = self.cross_norm(d_output + q_ctx)
        # ===== MoE fusion =====
        moe_input = torch.cat([d_output, code_h], dim=-1)
        moe_out = self.moe_fusion(moe_input)
        fused_state = moe_out + p_embed_data

        output_hidden = self.out(fused_state)

        labels = target.reshape(-1)
        preds = output_hidden.reshape(-1)
        mask = labels > -0.9 # 创建一个掩码，过滤掉padding（填充）部分

        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        # 主损失函数：二分类交叉熵
        loss = loss_fn(preds[mask], labels[mask].float())

        # ===== Contrastive learning (UNCHANGED) =====
        # 生成两个扰动版本，计算对比学习损失
        # ？？？？这块的两个扰动版本是怎么回事？怎么生成的？
        z1 = self.model(p_embed_data, pa_embed_data, pertubed=True, eps=0.2)
        z2 = self.model(p_embed_data, pa_embed_data, pertubed=True, eps=0.2)
        cl_loss = self.get_cl_loss(z1, z2, target >= 0)

        if not return_output:  # 总损失 = 主损失 + 0.1 * 对比学习损失
            return loss.sum() + 0.1 * cl_loss, torch.sigmoid(preds), mask.sum()
        else:
            return (
                loss.sum() + 0.1 * cl_loss,
                torch.sigmoid(preds),
                mask.sum(),
                d_output,
            )


class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'DACE'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks * 2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pertubed=False, eps=0.1):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed
        # 如果要做对比学习，随机打乱批次，就加一点噪声
        if pertubed:
            x_shuffle_idx = torch.randperm(x.shape[0]).to(device)
            y_shuffle_idx = torch.randperm(y.shape[0]).to(device)

            x_shuffle = x[x_shuffle_idx]
            y_shuffle = y[y_shuffle_idx]

            x = x + F.normalize(x_shuffle, p=2, dim=-1) * eps
            y = y + F.normalize(y_shuffle, p=2, dim=-1) * eps

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
            if pertubed:
                # 如果要做对比学习，每层都加点噪声
                y_shuffle_idx = torch.randperm(y.shape[0]).to(device)
                y_shuffle = y[y_shuffle_idx]
                # random_noise = torch.randn_like(y).cuda()
                y = y + F.normalize(y_shuffle, p=2, dim=-1) * eps

        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
                if pertubed:
                    x_shuffle_idx = torch.randperm(x.shape[0]).to(device)
                    x_shuffle = x[x_shuffle_idx]
                    x = x + F.normalize(x_shuffle, p=2, dim=-1) * eps
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
                if pertubed:
                    x_shuffle_idx = torch.randperm(x.shape[0]).to(device)
                    x_shuffle = x[x_shuffle_idx]
                    x = x + F.normalize(x_shuffle, p=2, dim=-1) * eps
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block 多头注意力
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        # 层归一化和Dropout
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        '''
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        '''
        # 创建注意力掩码
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        # 多头注意力
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)
        # 残差链接+层归一化
        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        # 前馈神经网络
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature # 每个头的维度
        self.h = n_heads # 头数
        self.kq_same = kq_same # k和q是否相同
        # 线性变换层
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias) # 输出投影
        # 可学习的gamma参数（用于位置效应）
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

# 注意力计算函数，这是最复杂的部分
def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        #
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        # 计算位置距离效应。让近的位置更重要，远的位置不那么重要
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    # 应用位置效应到注意力分数
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # import ipdb; ipdb.set_trace()
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


def init(model):
    model = model.to(device)
    # ===== BePKT / 无难度文件时，跳过 warmup =====
    if not os.path.exists(f'data/{Config.dataset}/question_difficulty.npy'):
        return
    # 如果有题目难度数据，先用难度预测任务预热题目嵌入，
    embed_l = model.p_embed.weight.shape[1]
    diff_pred = nn.Sequential(
        nn.Linear(embed_l, embed_l),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(embed_l, 1)
    ).to(device)
    loss_func = nn.MSELoss()
    # question difficulty warmup
    # 防止重复预热
    if hasattr(model, "warmup"):
        return
    model.warmup = True
    nn.init.normal_(model.p_embed.weight, mean=0, std=0.1)
    # 训练难度预测任务
    params = nn.ModuleList([diff_pred, model.p_embed])
    optimizer = torch.optim.Adam(params.parameters(), lr=0.001)
    pid_difficulty_labels = torch.FloatTensor(np.load(f'data/{Config.dataset}/question_difficulty.npy')).to(device)

    for epoch in range(50):
        model.train()
        p = diff_pred(model.p_embed.weight[1:]).reshape(-1)
        mse_loss = loss_func(p, pid_difficulty_labels)
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
    # 冻结题目嵌入，让它不再更新
    model.p_embed.weight.requires_grad = False  # True


