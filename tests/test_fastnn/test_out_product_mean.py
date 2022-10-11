import torch
from fastfold.model.fastnn.ops import OutProductMean as FastOutProductMean
from torch import nn
from typing import Optional
from test_basic_ops import Linear
import fastfold
from fastfold.model.fastnn.ops import set_chunk_size


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z)

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None,
                inplace_safe: bool = False,
    ) -> torch.Tensor:
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) 
        a = a * mask
        
        b = self.linear_2(ln) 
        b = b * mask

        del ln

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        if(inplace_safe):
            outer /= norm
        else:
            outer = outer / norm

        return outer


def test_out_product_mean():
    fastfold.distributed.init_dap()
    
    msa_len = 20
    seq_len = 30
    dim_m = 32
    dim_z = 64
    hidden = 16
    
    fast_opm = FastOutProductMean(n_feat=dim_m, n_feat_out=dim_z, n_feat_proj=hidden).cuda()
    opm = OuterProductMean(c_m=dim_m, c_z=dim_z, c_hidden=hidden).cuda()
    fast_opm.linear_a.weight = opm.linear_1.weight
    fast_opm.linear_a.bias = opm.linear_1.bias
    fast_opm.linear_b.weight = opm.linear_2.weight
    fast_opm.linear_b.bias = opm.linear_2.bias
    fast_opm.o_linear.weight = opm.linear_out.weight
    fast_opm.o_linear.bias = opm.linear_out.bias

    m = torch.randn((1, msa_len, seq_len, dim_m)).cuda()
    m_mask = torch.ones((1, msa_len, seq_len)).cuda()
    m_mask[:, :, -5:] = 0
    z = torch.zeros((1, seq_len, seq_len, dim_z)).cuda()

    out1 = fast_opm(m, m_mask, z)
    out2 = opm(m, m_mask)
    assert torch.allclose(out1, out2, atol=1e-6)

    set_chunk_size(1)
    out1 = fast_opm(m, m_mask, z)
    out2 = opm(m, m_mask)
    assert torch.allclose(out1, out2, atol=1e-6)

    out1 = opm(m, m_mask)
    out2 = fast_opm.inplace(m, m_mask, [z])[0]
    assert torch.allclose(out1, out2, atol=1e-6)

if __name__ == "__main__":
    test_out_product_mean()
