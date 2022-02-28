import torch.nn as nn

from fastfold.distributed.comm_async import All_to_All_Async, All_to_All_Async_Opp
from fastfold.model import MSAStack, OutProductMean, PairStack


class Evoformer(nn.Module):

    def __init__(self, d_node=256, d_pair=128):
        super(Evoformer, self).__init__()

        self.msa_stack = MSAStack(d_node, d_pair, p_drop=0.15)
        self.communication = OutProductMean(n_feat=d_node, n_feat_out=d_pair, n_feat_proj=32)
        self.pair_stack = PairStack(d_pair=d_pair)

    def forward(self, node, pair, node_mask, pair_mask):
        node = self.msa_stack(node, pair, node_mask)
        pair = pair + self.communication(node, node_mask)
        node, work = All_to_All_Async.apply(node, 1, 2)
        pair = self.pair_stack(pair, pair_mask)
        node = All_to_All_Async_Opp.apply(node, work, 1, 2)
        return node, pair