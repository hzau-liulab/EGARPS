import torch
import dgl
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence
import torch.nn.functional as F


class Graph_GINE(nn.Module):
    def __init__(self,apply_func):
        super(Graph_GINE,self).__init__()
        self.conv=dglnn.GINConv(apply_func,learn_eps=True,aggregator_type='mean')
        
    def forward(self,inputs):
        g,node_feat,coord_feat,edge_feat=inputs
        hx=self.conv(g,node_feat)
        return g,hx,coord_feat,edge_feat

class Graph_EGNN(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,edge_dim=0):
        super(Graph_EGNN,self).__init__()
        self.conv=dglnn.EGNNConv(in_dim,hidden_dim,out_dim,edge_dim)
        
    def forward(self,inputs):
        g,node_feat,coord_feat,edge_feat=inputs
        hx,coord=self.conv(g,node_feat,coord_feat,edge_feat)
        return g,hx,coord,edge_feat

class GRU(nn.Module):
    def __init__(self,in_size,hidden_size,out_size,num_layers=1):
        super(GRU,self).__init__()
        self.gru=nn.GRU(in_size,hidden_size,num_layers,batch_first=True)
        
    def forward(self,x):
        x=pack_sequence(x,enforce_sorted=False)
        out,hn=self.gru(x)
        out=unpack_sequence(out)
        return out,hn

class ContextAttention(nn.Module):
    def __init__(self, feature_dim):
        super(ContextAttention, self).__init__()
        
        self.W = nn.Linear(feature_dim, feature_dim) 
        self.W_prime = nn.Linear(feature_dim, feature_dim)
        
        self.w = nn.Linear(feature_dim, 1)
        self.tanh = nn.Tanh()
        
        self.w_p=nn.Linear(feature_dim,1)
        
    def forward(self, f_r, f_r_prime):
        q = self.W(f_r)
        k = self.W_prime(f_r_prime)
        
        q_expand=q.unsqueeze(1).expand(-1,k.shape[0],-1)
        alpha = self.w(self.tanh(q_expand + k))
        alpha = torch.softmax(alpha, dim=1)
        
        context = torch.sum(alpha * f_r_prime, dim=1)
        alpha_p=torch.softmax(self.w_p(context),dim=0)
        pool=torch.sum(alpha_p*context,dim=0,keepdim=True)
        
        return pool

class Attention(nn.Module):
    def __init__(self, embed_dim, latent_dim):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        self.attn_weight = nn.Linear(embed_dim, latent_dim)
        
        self.relu=nn.ReLU()
        
    def forward(self, primary, secondary):
        
        # primary: (N, embed_dim)
        # secondary: (M, embed_dim)
        
        N = primary.size(0)
        M = secondary.size(0)
        
        # Attention scores
        attn_scores = torch.matmul(self.attn_weight(primary), self.attn_weight(secondary).transpose(0,1)) #(N,M)
        
        attn_scores = self.relu(attn_scores)
        
        # attn_scores = attn_scores / torch.sqrt(torch.tensor(self.latent_dim).float())
        
        # attn_scores = nn.Softmax(dim=1)(attn_scores)  #(N,M)  
        
        attn_scores = F.normalize(attn_scores, p=2, dim=1)
        
        # Context vector
        context_vector = torch.matmul(attn_scores, secondary) #(N, embed_dim)
        
        return context_vector


class Graph_Model(nn.Module):
    def __init__(self, in_feats1=1992, in_feats2=1992, edge_feats=12, num_layers=2, knn=12):
        super(Graph_Model, self).__init__()
        
        # self.layers = nn.Sequential(*[
        #     Graph_EGNN(in_feats1, hidden_dim=(in_feats1+in_feats2+edge_feats)*1, out_dim=in_feats1+in_feats2, edge_dim=edge_feats)]+
        #     [Graph_EGNN(in_feats1+in_feats2, hidden_dim=(in_feats1+in_feats2+edge_feats)*1, out_dim=in_feats1+in_feats2, edge_dim=edge_feats) for _ in range(num_layers-1)]
        # )
        
        # self.maxpool=dglnn.MaxPooling()
        # self.weightsumpool=dglnn.WeightAndSum(in_feats1+in_feats2)
        
        self.knn=knn
        
        all_dim=in_feats1*4
        self.classify = nn.Sequential(
            nn.Linear(all_dim,all_dim),
            nn.ReLU(),
            nn.Linear(all_dim, int(all_dim/2)),
            # nn.LayerNorm(int(all_dim/2)),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(int(all_dim/2),1),
            )
        
        self.gru1=GRU(in_size=in_feats1,hidden_size=in_feats1,out_size=in_feats1)
        self.gru2=GRU(in_size=in_feats1,hidden_size=in_feats1,out_size=in_feats1)
        
        self.egnn=dglnn.EGNNConv(in_feats1,in_feats1,in_feats1,edge_feats)
        
        self.egnn1=dglnn.EGNNConv(in_feats1,in_feats1,in_feats1,edge_feats)
        self.egnn2=dglnn.EGNNConv(in_feats1,in_feats1,in_feats1,edge_feats)
        
        self.gap=dglnn.GlobalAttentionPooling(nn.Linear(in_feats1,1))
        
        # self.gap1=dglnn.GlobalAttentionPooling(nn.Linear(in_feats1,1))
        # self.gap2=dglnn.GlobalAttentionPooling(nn.Linear(in_feats1,1))
        
        # self.gap_intra=dglnn.GlobalAttentionPooling(nn.Linear(in_feats1*2,1),nn.Linear(in_feats1*2,in_feats1))
        # self.gap_inter=dglnn.GlobalAttentionPooling(nn.Linear(in_feats1,1))
        
        self.cap=ContextAttention(in_feats1)
        
        self.att1=Attention(in_feats1*2, in_feats1*2)
        self.att2=Attention(in_feats1*2, in_feats1*2)

    def forward(self, g):
        inter_g,pro_g,rna_g=g
        
        g,hx,ex,coord=self._get_feats(inter_g)
        hxe,_=self.egnn(g,hx,coord,ex)
        # inter_g.ndata['h']=hxe
        # inter_he=dgl.mean_nodes(inter_g,'h')
        # inter_g.ndata.pop('h')
        inter_he=self.gap(g,hxe)
        
        inter_g.ndata['h']=hx
        bgs=dgl.unbatch(inter_g)
        hlist=[self._get_pro_rna_node_feat(bg) for bg in bgs]
        pro_h=self.gru1([x[0] for x in hlist])[0]
        rna_h=self.gru2([x[1] for x in hlist])[0]
        pro_h_su=pro_h
        rna_h_su=rna_h
        pro_h=torch.cat([torch.mean(x,dim=0,keepdim=True) for x in pro_h],dim=0)
        rna_h=torch.cat([torch.mean(x,dim=0,keepdim=True) for x in rna_h],dim=0)
        # intra_hg=torch.cat((pro_h,rna_h),dim=1)
        
        g,hx,ex,coord=self._get_feats(pro_g)
        hxe,_=self.egnn1(g,hx,coord,ex)
        pro_h_se=self._split_batch_feat(pro_g.batch_num_nodes(), hxe)
        pro_g.ndata['h']=hxe
        pro_he=dgl.mean_nodes(pro_g,'h')
        pro_g.ndata.pop('h')
        # pro_he=self.gap1(g,hxe)
        
        g,hx,ex,coord=self._get_feats(rna_g)
        if ex != None:
            hxe,_=self.egnn2(g,hx,coord,ex)
        else:
            hxe=hx
        rna_h_se=self._split_batch_feat(rna_g.batch_num_nodes(), hxe)
        rna_g.ndata['h']=hxe
        rna_he=dgl.mean_nodes(rna_g,'h')
        rna_g.ndata.pop('h')
        # rna_he=self.gap2(g,hxe)
        
        # intra_he=torch.cat((pro_he,rna_he),dim=1)
        
        # # h=torch.cat((inter_he,intra_hg,intra_he),dim=1)
        # intra_h=torch.stack((intra_hg,intra_he),dim=1)
        # intra_h=self.gap_intra(dgl.batch([dgl.rand_graph(2,0,device=intra_h.device) for i in range(intra_h.shape[0])]),
        #                 intra_h.view(-1,intra_h.shape[-1]))
        # h=torch.stack((inter_he,intra_h),dim=1)
        # h=self.gap_inter(dgl.batch([dgl.rand_graph(2,0,device=h.device) for i in range(h.shape[0])]),
        #                 h.view(-1,h.shape[-1]))
        
        intra_h_g=torch.vstack([self.cap(torch.vstack((pro_h[i],rna_h[i],pro_he[i],rna_he[i])),
                  torch.vstack((pro_h[i],rna_h[i],pro_he[i],rna_he[i]))
                  ) 
                      for i in range(pro_h.shape[0])])
        
        pro_h_s=[torch.cat((pro_h_su[i],pro_h_se[i]),dim=1) 
                  for i in range(len(pro_h_su))]
        rna_h_s=[torch.cat((rna_h_su[i],rna_h_se[i]),dim=1) 
                  for i in range(len(rna_h_su))]
        intra_h_l=[torch.mean(
            torch.cat((self.att1(pro_h_s[i],rna_h_s[i]),self.att2(rna_h_s[i],pro_h_s[i])),
                              dim=0),dim=0,keepdim=True) 
                    for i in range(len(pro_h_s))]
        intra_h_l=torch.cat(intra_h_l,dim=0)
        
        h=torch.cat((inter_he,intra_h_g,intra_h_l),dim=1)
        # h=torch.cat((inter_he,intra_h_l),dim=1)
        
        h_o=self.classify(h)
        return h,h_o
    
    def _split_batch_feat(self,lens,feat):
        insert_value=torch.tensor([0],device='cuda')
        lens=torch.cumsum(lens,dim=0)
        lens=torch.cat((insert_value,lens),dim=0)
        
        bf=[feat[lens[i-1]:lens[i],:] for i in range(1,lens.shape[0])]
        return bf
    
    def _get_feats(self,g):
        hx=g.ndata.pop('x')
        #str[:,list(range(1946,1956))+list(range(1958,1992))]
        #seq[:,list(range(1946))+[1956,1957]]
        ex=g.edata.pop('x') if 'x' in g.edata else None
        #ex=None
        coord=g.ndata.pop('coord')
        return g,hx,ex,coord
    
    def _get_pro_rna_node_feat(self,g):
        moltype=g.ndata.pop('moltype')
        moltype=moltype.int()
        h=g.ndata.pop('h')
        pro_h=h[moltype==0]
        rna_h=h[moltype==1]
        return pro_h,rna_h
    
    def _get_intra_g(self,g):
        moltype=g.ndata.pop('moltype')
        moltype=moltype.int()
        
        h=g.ndata.pop('h')
        pro_h=h[moltype==0]
        rna_h=h[moltype==1]
        
        coord=g.ndata.pop('coord')
        pro_coord=coord[moltype==0]
        rna_coord=coord[moltype==1]
        
        pro_g=dgl.knn_graph(pro_coord,self.knn,exclude_self=True)
        pass
    
    def _avepool(self,g,res_feats,nt_feats):
        def _unbatch(nodes,num_nodes):
            nodes=torch.unique(nodes)
            num_nodes=torch.cumsum(num_nodes,dim=0)
            num_nodes=torch.cat((torch.tensor([0],device='cuda'),num_nodes))
            x=[nodes[torch.gt(nodes,num_nodes[i]-1) & torch.lt(nodes,num_nodes[i+1])] for i in range(len(num_nodes)-1)]
            return x
        
        res_num_nodes=g.batch_num_nodes('res')
        nt_num_nodes=g.batch_num_nodes('nt')
        res_nodes,nt_nodes=g.edges(etype='res2nt')
        res_ave=torch.cat([torch.mean(res_feats[x,:],dim=0,keepdim=True) 
                              for x in _unbatch(res_nodes,res_num_nodes)],dim=0)
        nt_ave=torch.cat([torch.mean(nt_feats[x,:],dim=0,keepdim=True) 
                              for x in _unbatch(nt_nodes,nt_num_nodes)],dim=0)
        return res_ave,nt_ave

# class Graph_Model(nn.Module):
#     def __init__(self, in_feats1, in_feats2, edge_feats, num_layers=2):
#         super(Graph_Model, self).__init__()
        
#         all_dim=int(in_feats1/2)
#         self.classify = nn.Sequential(
#             nn.Linear(all_dim*2, all_dim),
#             nn.RReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(all_dim, int(all_dim/2)),
#             nn.RReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(int(all_dim/2),1),
#             )

#     def forward(self, g):
#         g,hx,ex,coord=self._get_feats(g)
        
#         # average pool
#         g.ndata['h']=hx
#         h=dgl.mean_nodes(g,'h')
        
#         h=self.classify(h)
#         return h
    
#     def _get_feats(self,g):
#         hx=g.ndata.pop('x')
#         ex=g.edata.pop('x') if 'x' in g.edata else None
#         coord=g.ndata.pop('coord')
#         return g,hx,ex,coord