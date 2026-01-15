import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CodeSequenceTransformerEncoder(nn.Module):
    def __init__(
        self,
        code_emb_dim,
        d_model,
        n_heads=8,
        n_layers=2,
        d_ff=2048,
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = code_emb_dim * 2
        self.d_model = d_model

        self.input_proj = nn.Linear(self.input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, code_emb, response, seq_len):
        B, T, _ = code_emb.size()
        device = code_emb.device

        # -------- anti-leakage shift --------
        response_hist = torch.zeros_like(response)
        response_hist[:, 1:] = response[:, :-1]
        response_hist = response_hist.unsqueeze(-1)

        code_right = code_emb * response_hist
        code_wrong = code_emb * (1.0 - response_hist)
        x = torch.cat([code_right, code_wrong], dim=-1)
        x = self.input_proj(x)

        # causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        ).bool()

        padding_mask = torch.arange(T, device=device)[None, :] >= seq_len[:, None]

        h = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # shift (ensure h_t uses < t)
        h_shift = torch.zeros_like(h)
        h_shift[:, 1:] = h[:, :-1]

        return h_shift
