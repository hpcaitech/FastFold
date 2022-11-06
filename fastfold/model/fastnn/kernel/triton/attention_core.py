import triton
import triton.language as tl

# CREDITS: Initially inspired by the Triton tutorial


@triton.jit
def _attention_core(Q, K, V, mask, bias, sm_scale, TMP, Out, stride_qz, stride_qh, stride_qm,
                    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh,
                    stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX,
                    BATCH, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    use_mask: tl.constexpr, use_bias: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointers to bias, mask
    if use_bias:
        batch_2 = Z // BATCH
        off_hz_bias = (off_hz // (batch_2 * H) * H) + (off_hz % H)
        offs_base_bias = off_hz_bias * (N_CTX * N_CTX) + offs_m[:, None] * N_CTX + offs_n[None, :]

    if use_mask:
        off_hz_mask = (off_hz // H)
        offs_base_mask = off_hz_mask * N_CTX

    # initialize pointer to m and l
    t_ptrs = TMP + off_hz * N_CTX + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q_load_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        load_mask = (start_n + offs_n)[:, None] < N_CTX

        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn, mask=load_mask, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale

        qk = tl.where(offs_m[:, None] >= N_CTX, float("-1e20"), qk)
        qk = tl.where((start_n + offs_n)[None, :] >= N_CTX, float("-1e20"), qk)

        if use_bias:
            bias_load_mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            bias_load_mask = tl.where(offs_m[:, None] >= N_CTX, 1., bias_load_mask)
            bias_load_mask = tl.where((start_n + offs_n)[None, :] >= N_CTX, 1., bias_load_mask)
            bias_data = tl.load(bias + offs_base_bias + start_n,
                                mask=(bias_load_mask == 0.),
                                other=0.)
            qk += bias_data

        if use_mask:
            mask_data = tl.load(mask + offs_base_mask + offs_n + start_n,
                                mask=(start_n + offs_n) < N_CTX,
                                other=0.)
            qk = tl.where(mask_data[None, :] == 0., float("-1e20"), qk)

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale, mask=(offs_m < N_CTX))
        acc_scale = tl.load(TMP + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M),
                            mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M) < N_CTX),
                            other=float(0.))  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        load_mask = (start_n + offs_n)[:, None] < N_CTX
        v = tl.load(v_ptrs + start_n * stride_vn, mask=load_mask, other=0.)
        p = p.to(Q.dtype.element_ty)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, l_i)
    # tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o

    out_store_mask = offs_m[:, None] < N_CTX
    tl.store(out_ptrs, acc, mask=out_store_mask)
