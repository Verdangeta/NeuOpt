import os
import torch
from torch import nn
import torch.multiprocessing as mp
from nets.graph_layers import MultiHeadEncoder, EmbeddingNet, MultiHeadPosCompat, kopt_Decoder
from utils import masked_dist_matrix
from RTD_Lite_TSP import RTD_Lite, prim_algo

def _run_rtd_lite(args):
    edge, tour, mst = args
    return RTD_Lite(edge, tour)(mst)[2]

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
class Actor(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_layers,
                 normalization,
                 v_range,
                 seq_length,
                 k,
                 with_RNN,
                 with_feature1,
                 with_feature3,
                 with_simpleMDP,
                 with_RTDL
                 ):
        super(Actor, self).__init__()

        problem_name = problem.NAME
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.seq_length = seq_length                
        self.k = k
        self.with_RNN = with_RNN
        self.with_feature1 = with_feature1
        self.with_feature3 = with_feature3
        self.with_simpleMDP = with_simpleMDP
        self.with_RTDL = with_RTDL
        
        if problem_name == 'tsp':
            self.node_dim = 2
        elif problem_name == 'cvrp':
            self.node_dim = 8 if self.with_feature1 else 6
        else:
            raise NotImplementedError()
            
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim,
                            self.seq_length)
        
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor, 
                                self.embedding_dim, 
                                self.hidden_dim,
                                number_aspect = 2,
                                normalization = self.normalization
                                )
            for _ in range(self.n_layers)))

        self.pos_encoder = MultiHeadPosCompat(self.n_heads_actor, 
                                self.embedding_dim, 
                                self.hidden_dim, 
                                )
        
        self.decoder = kopt_Decoder(self.n_heads_actor, 
                                    input_dim = self.embedding_dim, 
                                    embed_dim = self.embedding_dim,
                                    v_range = self.range,
                                    k = self.k,
                                    with_RNN = self.with_RNN,
                                    with_feature3 = self.with_feature3,
                                    simpleMDP = self.with_simpleMDP
                                    )

        print('# params in Actor', self.get_parameter_number())

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def precompute_rtdl_mst(self, batch):
        coords = batch['coordinates']
        edge_len = torch.cdist(coords, coords, p=2)
        mst_list = []
        for i in range(len(edge_len)):
            _, edge_idx, edge_w = prim_algo(edge_len[i].cpu())
            edge_idx = edge_idx[edge_w.argsort()]
            edge_w = edge_w[edge_w.argsort()]
            mst_list.append((edge_idx, edge_w))
        return mst_list

    def compute_rtdl_features(self, batch, solution, mst_list=None):
        coords = batch['coordinates']
        edge_len = torch.cdist(coords, coords, p=2)
        tour_edge_len = masked_dist_matrix(solution, edge_len)
        rtdl_features = torch.zeros_like(tour_edge_len)
        # ctx = mp.get_context("spawn")
        # with ctx.Pool(processes=min(len(edge_len), os.cpu_count())) as pool:
            # results = pool.map(_run_rtd_lite,
            #                    [(edge_len[i], tour_edge_len[i]) for i in range(len(edge_len))])
        results = [_run_rtd_lite((edge_len[i], tour_edge_len[i], None if mst_list is None else mst_list[i])) for i in range(len(edge_len))]
        for i, feat in enumerate(results):
            rtdl_features[i] = feat
        return rtdl_features

    def forward(self, problem, batch, x_in, solution, context, context2,last_action, fixed_action = None, require_entropy = False, to_critic = False, only_critic  = False, rtdl_features=None):
        # the embedded input x
        bs, gs, in_d = x_in.size()
        if self.with_RTDL and rtdl_features is None:
            rtdl_features = self.compute_rtdl_features(batch, solution)

        if problem.NAME == 'cvrp':
            
            visited_time, to_actor = problem.get_dynamic_feature(solution, batch, context)
            if self.with_feature1:
                x_in = torch.cat((x_in, to_actor), -1)
            else:
                x_in = torch.cat((x_in, to_actor[:,:,:-2]), -1)
            del context, to_actor

        elif problem.NAME == 'tsp':
            visited_time = problem.get_order(solution, return_solution = False)
        else: 
            raise NotImplementedError()
            
        h_embed, h_pos = self.embedder(x_in, solution, visited_time)
        aux_scores = self.pos_encoder(h_pos)
        
        h_em_final, _ = self.encoder(h_embed, aux_scores)
        
        if only_critic:
            return (h_em_final)
        
        action, log_ll, entropy = self.decoder(problem,
                                               h_em_final,
                                               solution,
                                               context2,
                                               visited_time,
                                               last_action,
                                               rtdl_features,
                                               fixed_action = fixed_action,
                                               require_entropy = require_entropy)
        
        # assert (visited_time == visited_time_clone).all()
        if require_entropy:
            return action, log_ll, (h_em_final) if to_critic else None, entropy
        else:
            return action, log_ll, (h_em_final) if to_critic else None
