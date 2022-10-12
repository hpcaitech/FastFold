import torch
import fastfold
from fastfold.model.fastnn.ops import OutProductMean as FastOutProductMean, set_chunk_size
from fastfold.model.nn.outer_product_mean import OuterProductMean


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

    out = fast_opm(m, m_mask, z)
    
    out_fast = opm(m, m_mask)
    assert torch.allclose(out, out_fast, atol=1e-6)

    set_chunk_size(1)
    out_fast = opm(m, m_mask)
    assert torch.allclose(out, out_fast, atol=1e-6)

    out_fast = fast_opm.inplace(m, m_mask, [z])[0]
    assert torch.allclose(out, out_fast, atol=1e-6)


if __name__ == "__main__":
    test_out_product_mean()
