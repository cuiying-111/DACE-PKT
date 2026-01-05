import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CodeSequenceGRUEncoder(nn.Module):
    """
    Leakage-safe Code Sequence Encoder for PKT / DACE.

    Design guarantees:
    1) Explicitly receives response (0/1)
    2) Internally constructs history-only response (shifted)
    3) Hidden state at time t NEVER uses r_t

    Input:
        code_emb : (B, T, D)
        response : (B, T), r_t
        seq_len  : (B,), true submission lengths

    Output:
        h : (B, T, hidden_dim)
            h[:, t] encodes coding ability BEFORE step t
    """

    def __init__(self, code_emb_dim, hidden_dim):
        super(CodeSequenceGRUEncoder, self).__init__()

        self.code_emb_dim = code_emb_dim
        self.hidden_dim = hidden_dim

        # response-integrated embedding: [c ⊕ 0] or [0 ⊕ c]
        self.input_dim = code_emb_dim * 2

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

    def forward(self, code_emb, response, seq_len):
        """
        Parameters
        ----------
        code_emb : Tensor
            (B, T, D)

        response : Tensor
            (B, T), r_t (will be shifted internally)

        seq_len : Tensor
            (B,), real sequence length (no padding)
        """

        B, T, D = code_emb.size()
        device = code_emb.device

        # --------------------------------------------------
        # 1. Build history-only response (ANTI-LEAKAGE)
        #    response_hist[:, t] = r_{t-1}
        # --------------------------------------------------
        response_hist = torch.zeros_like(response)
        response_hist[:, 1:] = response[:, :-1]
        response_hist = response_hist.unsqueeze(-1)  # (B, T, 1)

        # --------------------------------------------------
        # 2. Response-integrated code embedding
        #    If r_{t-1}=1: [c_t ⊕ 0]
        #    If r_{t-1}=0: [0 ⊕ c_t]
        # --------------------------------------------------
        code_right = code_emb * response_hist
        code_wrong = code_emb * (1.0 - response_hist)

        code_input = torch.cat([code_right, code_wrong], dim=-1)
        # (B, T, 2D)

        # --------------------------------------------------
        # 3. Pack padded sequence (safe temporal modeling)
        # --------------------------------------------------
        packed_input = pack_padded_sequence(
            code_input,
            seq_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # --------------------------------------------------
        # 4. GRU forward
        # --------------------------------------------------
        packed_output, _ = self.gru(packed_input)

        # --------------------------------------------------
        # 5. Unpack (padding positions are zero)
        # --------------------------------------------------
        h, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=T
        )

        h_shift = torch.zeros_like(h)
        h_shift[:, 1:] = h[:, :-1]

        return h_shift
