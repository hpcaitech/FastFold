import argparse
import os

import torch
import torch.nn as nn

from fastfold.distributed import init_dap
from fastfold.model.fastnn import Evoformer


def main():

    parser = argparse.ArgumentParser(description='Evoformer Standalone Perf Benchmark')
    parser.add_argument("--dap-size", default=1, type=int, help='batch size')
    parser.add_argument('--batch-size', default=1, type=int, help='batch size')
    parser.add_argument('--msa-length', default=132, type=int, help='Sequence Length of MSA')
    parser.add_argument('--res-length',
                        default=256,
                        type=int,
                        help='Sequence Length of Residues')
    parser.add_argument('--trials', default=50, type=int, help='Number of Trials to Execute')
    parser.add_argument('--warmup-trials', default=5, type=int, help='Warmup Trials to discard')
    parser.add_argument('--layers',
                        default=12,
                        type=int,
                        help='Evoformer Layers to Execute')
    parser.add_argument('--cm', default=256, type=int, help='MSA hidden dimension')
    parser.add_argument('--cz', default=128, type=int, help='Pair hidden dimension')
    parser.add_argument('--heads', default=8, type=int, help='Number of Multihead Attention heads')
    parser.add_argument('--openfold',
                        action='store_true',
                        help='Benchmark with Evoformer Implementation from OpenFold.')
    parser.add_argument('--fwd', action='store_true', help='Only execute Fwd Pass.')
    parser.add_argument('--prof', action='store_true', help='run with profiler.')

    args = parser.parse_args()

    init_dap(args.dap_size)

    precision = torch.bfloat16
    if args.dap_size > 1:
        # (PyTorch issue) Currently All2All communication does not support the Bfloat16 datatype in PyTorch
        precision = torch.float16

    if not torch.cuda.is_available():
        raise NotImplementedError('Running on CPU is not supported')

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.openfold:
        from openfold.model.evoformer import EvoformerBlock

        class OpenFoldEvoformer(nn.Module):

            def __init__(self, d_node, d_pair):
                super(OpenFoldEvoformer, self).__init__()
                self.d_node = d_node
                self.d_pair = d_pair

                self.c_hidden_msa_att = int(d_node / 8)
                self.c_hidden_pair_att = int(d_pair / 8)

                self.EvoformerBlock = EvoformerBlock(c_m=d_node,
                                                     c_z=d_pair,
                                                     c_hidden_msa_att=self.c_hidden_msa_att,
                                                     c_hidden_opm=self.c_hidden_msa_att,
                                                     c_hidden_mul=self.d_pair,
                                                     c_hidden_pair_att=self.c_hidden_pair_att,
                                                     no_heads_msa=8,
                                                     no_heads_pair=4,
                                                     transition_n=4,
                                                     msa_dropout=0.15,
                                                     pair_dropout=0.25,
                                                     inf=1e9,
                                                     eps=1e-10)

            def forward(self, node, pair, node_mask, pair_mask):
                node, pair = self.EvoformerBlock(node, pair, node_mask, pair_mask)
                return node, pair

    attn_layers = []
    for idx in range(0, args.layers):
        if args.openfold:
            attn_layers.append(OpenFoldEvoformer(d_node=args.cm, d_pair=args.cz))
        else:
            attn_layers.append(Evoformer(d_node=args.cm, d_pair=args.cz))
        attn_layers[idx].cuda()
        attn_layers[idx].to(dtype=precision)

    start_evt_fwd = []
    start_evt_bwd = []
    stop_evt_bwd = []
    for recorded_trial in range(0, args.trials):
        start_evt_fwd.append(torch.cuda.Event(enable_timing=True))
        start_evt_bwd.append(torch.cuda.Event(enable_timing=True))
        stop_evt_bwd.append(torch.cuda.Event(enable_timing=True))

    inputs_node = torch.randn(args.batch_size,
                              args.msa_length // args.dap_size,
                              args.res_length,
                              args.cm,
                              dtype=precision,
                              device=torch.device("cuda")).requires_grad_(True)
    inputs_pair = torch.randn(args.batch_size,
                              args.res_length // args.dap_size,
                              args.res_length,
                              args.cz,
                              dtype=precision,
                              device=torch.device("cuda")).requires_grad_(True)
    node_mask = torch.ones((args.batch_size, args.msa_length, args.res_length),
                           dtype=precision,
                           device=torch.device("cuda")).requires_grad_(False)
    pair_mask = torch.ones((args.batch_size, args.res_length, args.res_length),
                           dtype=precision,
                           device=torch.device("cuda")).requires_grad_(False)
    grads_node = torch.randn_like(inputs_pair)

    if args.prof:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1,
                                             warmup=args.warmup_trials,
                                             active=args.trials,
                                             repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fastfold'),
            profile_memory=False,
            record_shapes=False,
            with_stack=False)
        prof.start()

    for trial in range(0, args.trials + args.warmup_trials):
        layer_inputs = inputs_node, inputs_pair
        evt_idx = trial - args.warmup_trials

        torch.distributed.barrier()
        torch.cuda.synchronize()

        if evt_idx >= 0:
            start_evt_fwd[evt_idx].record()

        for lyr_idx in range(0, args.layers):
            layer_inputs = attn_layers[lyr_idx].forward(*layer_inputs, node_mask, pair_mask)

        torch.cuda.synchronize()

        if evt_idx >= 0:
            start_evt_bwd[evt_idx].record()

        if not args.fwd:
            layer_inputs[1].backward(grads_node)

        if evt_idx >= 0:
            stop_evt_bwd[evt_idx].record()

        if args.prof:
            prof.step()

    if args.prof:
        prof.stop()

    torch.distributed.barrier()
    torch.cuda.synchronize()
    elapsed_time_fwd = 0.0
    elapsed_time_bwd = 0.0
    for evt_idx in range(0, args.trials):
        elapsed_time_fwd += start_evt_fwd[evt_idx].elapsed_time(start_evt_bwd[evt_idx])
        elapsed_time_bwd += start_evt_bwd[evt_idx].elapsed_time(stop_evt_bwd[evt_idx])

    print("[ MSA Attn ] Input: {:4d}, {:4d}, {:4d}, ({:4d} {:4d}) Fwd Time / Layer: {:.3f} ms Bwd Time / Layer: {:.3f} ms".format(
        args.batch_size, args.msa_length, args.res_length,     \
        args.cm, args.cz,                                      \
        elapsed_time_fwd / ( args.trials * args.layers ),      \
        elapsed_time_bwd / ( args.trials * args.layers )))


if __name__ == '__main__':
    main()
