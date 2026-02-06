from dgl.nn.pytorch import GraphConv
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import torch as th

class GraphConvEdgeWeight(GraphConv):
    def forward(self, graph, feat, weight=None, edge_weights=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            # print(f"Shape of feat_src: {feat_src.shape}")
            # print(f"Shape of feat_dst: {feat_dst.shape}")

            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm
                # print(f"Shape of normalized feat_src: {feat_src.shape}")

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            # print(f"Shape of weight: {weight.shape if weight is not None else 'None'}")

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                    # print(f"Shape of feat_src after weight multiplication: {feat_src.shape}")
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_u(u='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                # print(f"Shape of rst after aggregation: {rst.shape}")
                if weight is not None:
                    rst = th.matmul(rst, weight)
                    # print(f"Shape of rst after weight multiplication: {rst.shape}")

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm
                # print(f"Shape of rst after normalization: {rst.shape}")

            if self.bias is not None:
                rst = rst + self.bias
                # print(f"Shape of rst after adding bias: {rst.shape}")

            if self._activation is not None:
                rst = self._activation(rst)
                # print(f"Shape of rst after activation: {rst.shape}")

            return rst
