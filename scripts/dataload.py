import os,re
os.environ['DGLBACKEND'] = 'pytorch'
import pickle
import dgl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from Models_res import Graph_Model
import random
from dgl.dataloading import GraphDataLoader

import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from datapre import STD_ATOM_NAMES, compute_rel_geom_feats, ENCODE_RES
from atom import AtomPair


class HeteroGraphDataset(dgl.data.DGLDataset):
    train_seq_embed=dict()
    test_seq_embed=dict()
    # installing_feat_dir=''
    
    onehot_pro=21
    onehot_rna=5
    
    ss_pro=3
    ss_rna=7
    
    position_pro=1
    position_rna=1
    
    torsion_pro=4
    torsion_rna=12
    
    rsa_pro=1
    rsa_rna=1
    
    lap_pro=5
    lap_rna=5
    
    esmif_pro=0
    esmif_rna=0
    
    pocketness_pro=3
    pocketness_rna=3
    
    topo_pro=4
    topo_rna=4
    
    usr_pro=3
    usr_rna=3
    
    def __init__(self,hetero_graphs,train=True,N=50,num_neg_samples=500,rmsd=None):#k->layer nodes included, rmsd->dict
        self.data_len=len(hetero_graphs)
        # self._load_graph(hetero_graphs)
        self.hetero_graphs=hetero_graphs
        self.N=N
        self.train=train
        self.rmsd=rmsd
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        tags,graphs,targets,tag_atom_resindex=self._load_graph([self.hetero_graphs[idx]])[0]
        if self.train:
            sample_weights=self._get_rmsds(tags, self.rmsd)
            # rand_idx=random.sample(range(1,len(tags)),self.N)
            rand_idx=random.choices(range(1,len(tags)),weights=sample_weights,k=self.N) #exist same value
            # rand_idx=random.sample(sample_weights[:20],self.N)
            rand_idx.insert(0,0)
            # gs=[graphs[i] for i in rand_idx]
            gs=[self._add_seq_embedding(graphs[i],tag_atom_resindex[tags[i]],tags[0]) for i in rand_idx]
            return dgl.batch(gs),targets[rand_idx,:],[tags[i] for i in rand_idx]
        else:
            graphs.pop(0)
            bgs=list()
            for i in range(0,len(graphs),self.N):
                gs=[self._add_seq_embedding(graphs[j],tag_atom_resindex[tags[j]],tags[0]) for j in range(i,i+self.N)]
                bgs.append(dgl.batch(gs))
            return bgs,targets,tags
    
    def _load_graph(self,graph):
        def _load(g):
            with open(g,'rb') as f:
                datadict=pickle.load(f)
            return datadict
        
        datadicts=[_load(g) for g in graph]
        return [self._get_targets_graphs(data,graphf) for data,graphf in zip(datadicts,graph)] #N*3*N
    
    def _add_seq_embedding(self,g,atom_res_index,tag):
        atom_num=g.ndata['atom_number_ori'].tolist()
        moltype=['protein' if int(x)==0 else 'rna' for x in g.ndata['moltype'].tolist()]
        seq_embedding=list()
        for x,y in zip(atom_num,moltype):
            pdb=tag+'_'+y+'_'+atom_res_index[x][0]
            res_num=int(atom_res_index[x][1])
            seq_embed=self.train_seq_embed[pdb][res_num] if self.train else self.test_seq_embed[pdb][res_num]
            if y=='protein':
                seq_embed=np.pad(seq_embed,(0,640),mode='constant', constant_values=0)
            else:
                seq_embed=np.pad(seq_embed,(1280,0),mode='constant', constant_values=0)
            seq_embedding.append(seq_embed)
        seq_embedding=np.row_stack(seq_embedding)
        seq_embedding=torch.tensor(seq_embedding,dtype=torch.float32)
        if 'x' in g.ndata:
            g.ndata['x']=torch.cat((g.ndata['x'],seq_embedding),dim=1)
        else:
            g.ndata['x']=seq_embedding
        return g
    
    def _prepare_g(self,g):
        mass_charge=g.ndata.pop('mass_charge') #2
        atom_type=g.ndata.pop('atom_type') #65
        atom_coord=g.ndata.pop('x_pred') #3
        surf_prox=g.ndata.pop('surf_prox') #1
        # dpx_cx=g.ndata.pop('dpx_cx_lap')[:,:2] #2
        # chain_id=g.ndata.pop('chain_id') #1
        # residue_number=g.ndata.pop('residue_number') #-1
        # moltype=g.ndata.pop('moltype') #-1
        
        pos_enc=g.edata.pop('pos_enc') # 1
        in_same_chain=g.edata.pop('in_same_chain') #1
        geom=g.edata.pop('rel_geom_feats') #12
        in_same_molecular=g.edata.pop('in_same_molecular') #3
        
        g.ndata['x']=torch.cat((atom_type,mass_charge,surf_prox),dim=1)
        g.ndata['coord']=atom_coord
        g.edata['x']=torch.cat((geom,in_same_chain,in_same_molecular),dim=1)
        return g
    
    def _get_targets_graphs(self,data,graphf):#dict
        tags=list()
        graphs=list()
        pos_tag=''
        tag_atom_resindex=dict() #used for add sequnce embeddings
        for k,v in data.items():
            if not re.match('complex20240520',k):
                pos_tag=k.replace('_fix','')
                pos_g=self._prepare_g(v[0]) if isinstance(v,tuple) else self._prepare_g(v)
                tag_atom_resindex[pos_tag]=v[1] if isinstance(v,tuple) else None
            else:
                tags.append(k.replace('_fix',''))
                graphs.append(self._prepare_g(v[0])) if isinstance(v,tuple) else self._prepare_g(v)
                tag_atom_resindex[k.replace('_fix','')]=v[1] if isinstance(v,tuple) else None
        if not pos_tag: #to check some not have interface pdbs
            pos_tag=os.path.basename(graphf).replace('.pkl','')
            pos_g=graphs[0]
            tag_atom_resindex[pos_tag]=tag_atom_resindex[tags[0]]

        tags.insert(0,pos_tag)
        graphs.insert(0,pos_g)
        targets=np.zeros((len(tags),1),dtype=int)
        targets[0,0]=1
        return tags,graphs,targets,tag_atom_resindex
    
    def _get_rmsds(self,tags,rmsddict):
        return [rmsddict[tags[0]][x] for x in tags[1:]]
    
class HeteroGraphDataset_resgraph(HeteroGraphDataset):
    def __init__(self,knn,external_feats,temp_dataset_dir,**kwargs): #int,list
        super(HeteroGraphDataset_resgraph,self).__init__(**kwargs)
        self.knn=knn
        self.external_feats=external_feats
        
        self.feat_embed=dict()
        for feat_type in self.external_feats:
            if feat_type=='embedding':
                continue
            with open(self.installing_feat_dir+feat_type+'.pkl','rb') as f:
                self.feat_embed[feat_type]=pickle.load(f)
        
        self.temp_dataset_dir=temp_dataset_dir
        
    def __getitem__(self, idx):
        tags,graphs,targets,tag_atom_resindex=self._load_graph([self.hetero_graphs[idx]])[0]
        
        if self.train:
            sample_weights=self._get_rmsds(tags, self.rmsd)
            # rand_idx=random.sample(range(1,len(tags)),self.N)
            rand_idx=random.choices(range(1,len(tags)),weights=sample_weights,k=self.N) #exist same value
            # rand_idx=random.sample(sample_weights[:20],self.N)
            rand_idx.insert(0,0)
            # gs=[graphs[i] for i in rand_idx]
            gs=[self._assign_node_feat(self._knn_graph(graphs[i],self.knn),tag_atom_resindex[tags[i]],tags[0],self.external_feats) for i in rand_idx]
            return dgl.batch(gs),targets[rand_idx,:],[tags[i] for i in rand_idx]
        else:
            if os.path.exists(os.path.join(self.temp_dataset_dir,tags[0]+'.pkl')):
                with open(os.path.join(self.temp_dataset_dir,tags[0]+'.pkl'),'rb') as f:
                    bgs,targets,tags=pickle.load(f)
            else:
                graphs.pop(0)
                bgs=list()
                for i in range(0,len(graphs),self.N):
                    gs=[self._assign_node_feat(self._knn_graph(graphs[j],self.knn),tag_atom_resindex[tags[j]],tags[0],self.external_feats) for j in range(i,i+self.N)]
                    bgs.append(dgl.batch(gs))
                
                with open(os.path.join(self.temp_dataset_dir,tags[0]+'.pkl'),'wb') as f:
                    pickle.dump((bgs,targets,tags),f)
            
            return bgs,targets,tags
    
    def _map_edges_feat(self,ori_g,g,node_id_map):#to map old graph edges feat to new graph edges
        orifeat=ori_g.edata['x']
        target=torch.stack(ori_g.edges(),dim=1);print(target.shape)
        query=node_id_map[torch.stack(g.edges(),dim=1)];print(query.shape)
        with open('./test.pkl','wb') as f:
            pickle.dump((node_id_map,target,query),f)
        print(node_id_map.shape)
        print(query[:,None].shape)
        print(torch.eq(query[:, None], target).shape)
        indices=torch.nonzero(torch.all(torch.eq(query[:, None], target), dim=2))[:, 1]
        print(indices.shape)
        feat=orifeat[indices]
        return feat
    
    def _knn_graph(self,g,knn=24,types='inter'):
        atom_type=g.ndata['x'][:,:65].int()
        atom_type=atom_type==1
        if types=='inter':
            res_index=atom_type[:,0]+atom_type[:,37] #preserve CA and C3'
        elif types=='protein':
            res_index=atom_type[:,0]
        elif types=='rna':
            res_index=atom_type[:,37]
        knn_graph = dgl.knn_graph(g.ndata['coord'][res_index], knn, exclude_self=True)
        
        # node_id_map=torch.nonzero(res_index).squeeze()
        
        knn_graph.ndata['coord']=g.ndata['coord'][res_index]
        knn_graph.ndata['atom_number_ori']=g.ndata['atom_number_ori'][res_index]
        knn_graph.ndata['moltype']=g.ndata['moltype'][res_index]
        knn_graph.ndata['residue_number']=g.ndata['residue_number'][res_index]
        if len(knn_graph.nodes())>1:
            if types=='inter':
                knn_graph.edata['x']=torch.cat((self._assign_edge_geom_feat(g,knn_graph),
                                                self._assign_edge_atom_feat(g,knn_graph)), 
                                               dim=1)
            else:
                knn_graph.edata['x']=self._assign_edge_geom_feat(g,knn_graph)
        knn_graph.ndata.pop('residue_number')
        return knn_graph
    
    def _assign_edge_geom_feat(self,ori_g,g):
        data={'x_coord':ori_g.ndata['coord'][:,0].tolist(),
              'y_coord':ori_g.ndata['coord'][:,1].tolist(),
              'z_coord':ori_g.ndata['coord'][:,2].tolist(),
              'chain_id':ori_g.ndata['chain_id'][:,0].int().tolist(),
              'residue_number':ori_g.ndata['residue_number'].tolist(),
              'insertion':'',
              'atom_name':[STD_ATOM_NAMES[x] 
                  for x in torch.where(ori_g.ndata['x'][:,:65].int()==1)[1].tolist()],
              }
        atoms_df=pd.DataFrame(data)
        rel_geom=compute_rel_geom_feats(g, atoms_df)
        return rel_geom
    
    def _assign_edge_atom_feat(self,ori_g,g):
        dist=torch.cdist(ori_g.ndata['coord'][ori_g.ndata['moltype']==0], 
                         ori_g.ndata['coord'][ori_g.ndata['moltype']==1])
        
        feat_dict=dict()
        indices=torch.arange(0,len(ori_g.ndata['residue_number']))
        for i in g.nodes()[:-1]:
            for j in g.nodes()[i+1:]:
                i_index=ori_g.ndata['residue_number']==g.ndata['residue_number'][i]
                j_index=ori_g.ndata['residue_number']==g.ndata['residue_number'][j]
                feat=torch.zeros((20,12))
                if ori_g.ndata['moltype'][i_index][0]==ori_g.ndata['moltype'][j_index][0]:
                    feat=feat.reshape(1,-1)
                else:
                    for ii in indices[i_index]:
                        for jj in indices[j_index]:
                            prpo=AtomPair(ENCODE_RES[ori_g.ndata['residue'][ii]],
                                          STD_ATOM_NAMES[ori_g.ndata['x'][ii,:65].int().tolist().index(1)],
                                          ENCODE_RES[ori_g.ndata['residue'][jj]],
                                          STD_ATOM_NAMES[ori_g.ndata['x'][jj,:65].int().tolist().index(1)],)
                            if None in prpo:
                                continue
                            if ori_g.ndata['moltype'][ii]==1:
                                ii-=dist.shape[0]
                            if ori_g.ndata['moltype'][jj]==1:
                                jj-=dist.shape[0]
                            feat[prpo[0],prpo[1]]=dist[ii,jj]
                    feat=feat.reshape(1,-1)
                feat_dict[(i.item(),j.item())]=feat
        # print(feat_dict.keys())
        atom_feat=list()
        for i,j in zip(g.edges()[0],g.edges()[1]):
            i=i.item()
            j=j.item()
            if (i,j) in feat_dict:
                atom_feat.append(feat_dict[(i,j)])
            else:
                atom_feat.append(feat_dict[(j,i)])
        atom_feat=torch.row_stack(atom_feat)
        return atom_feat
    
    def _install_external_feat(self,g,atom_res_index,tag,feat_type):
        atom_num=g.ndata['atom_number_ori'].tolist()
        moltype=['protein' if int(x)==0 else 'rna' for x in g.ndata['moltype'].tolist()]
        feat_embed=self.feat_embed[feat_type]
        
        if feat_type=='onehot':
            pro_pad=self.onehot_pro
            rna_pad=self.onehot_rna
        elif feat_type=='ss':
            pro_pad=self.ss_pro
            rna_pad=self.ss_rna
        elif feat_type=='backbone_torsion':
            pro_pad=self.torsion_pro
            rna_pad=self.torsion_rna
        elif feat_type=='position':
            pro_pad=self.position_pro
            rna_pad=self.position_rna
        elif feat_type=='rsa':
            pro_pad=self.rsa_pro
            rna_pad=self.rsa_rna
        elif feat_type=='lap':
            pro_pad=self.lap_pro
            rna_pad=self.lap_rna
        elif feat_type=='esmif':
            pro_pad=self.esmif_pro
            rna_pad=self.esmif_rna
        elif feat_type=='pocketness':
            pro_pad=self.pocketness_pro
            rna_pad=self.pocketness_rna
        elif feat_type=='topo':
            pro_pad=self.topo_pro
            rna_pad=self.topo_rna
        elif feat_type=='usr':
            pro_pad=self.usr_pro
            rna_pad=self.usr_rna
        
        feat=list()
        for x,y in zip(atom_num,moltype):
            pdb=tag+'_'+y+'_'+atom_res_index[x][0]
            res_num=int(atom_res_index[x][1])
            
            seq_embed=feat_embed['train'][pdb][res_num] if self.train else feat_embed['test'][pdb][res_num]
            
            if y=='protein':
                seq_embed=np.pad(seq_embed,(0,rna_pad),mode='constant', constant_values=0)
            else:
                seq_embed=np.pad(seq_embed,(pro_pad,0),mode='constant', constant_values=0)
            feat.append(seq_embed)
        
        feat=np.row_stack(feat)
        feat=torch.tensor(feat,dtype=torch.float32)
        
        if 'x' in g.ndata:
            g.ndata['x']=torch.cat((g.ndata['x'],feat),dim=1)
        else:
            g.ndata['x']=feat
        
        return g
    
    def _assign_node_feat(self,g,atom_res_index,tag,feat_types):#feat_type->list
        for feat_type in feat_types:
            if feat_type=='embedding':
                g=self._add_seq_embedding(g, atom_res_index, tag)
            else:
                g=self._install_external_feat(g, atom_res_index, tag, feat_type)
        
        g.ndata.pop('atom_number_ori')
        # g.ndata.pop('moltype')
        
        return g

class HeteroGraphDataset_resgraph_siamese(HeteroGraphDataset_resgraph):
    def __init__(self,installing_feat_dir,**kwargs):
        self.installing_feat_dir=installing_feat_dir
        super(HeteroGraphDataset_resgraph_siamese,self).__init__(**kwargs)
    
    def __getitem__(self,idx):
        tags,graphs,targets,tag_atom_resindex=self._load_graph([self.hetero_graphs[idx]])[0]
        if self.train:
            sample_weights=self._get_rmsds(tags,self.rmsd)
            
            # rand_idx=random.sample(range(1,len(tags)),1) #one oo sample
            rand_idx=random.choices(range(1,len(tags)),weights=sample_weights,k=1)
            
            rand_idx.insert(0,0)
            siamese_targets=[1] if float(sample_weights[rand_idx[-1]-1])<=4 else [0]
            siamese_targets=np.array(siamese_targets,dtype=int).reshape(-1,1)
            bce_targets=[1,0] #if float(sample_weights[rand_idx[-1]-1])<=4 else [1,0]
            bce_targets=np.array(bce_targets,dtype=int).reshape(-1,1)
            
            inter_gs=[self._assign_node_feat(self._knn_graph(graphs[i],self.knn),tag_atom_resindex[tags[i]],tags[0],self.external_feats) for i in rand_idx]
            pro_gs=[self._assign_node_feat(self._knn_graph(graphs[i],self.knn,'protein'),tag_atom_resindex[tags[i]],tags[0],self.external_feats) for i in rand_idx]
            rna_gs=[self._assign_node_feat(self._knn_graph(graphs[i],self.knn,'rna'),tag_atom_resindex[tags[i]],tags[0],self.external_feats) for i in rand_idx]
            return (dgl.batch(inter_gs),dgl.batch(pro_gs),dgl.batch(rna_gs)),bce_targets,[tags[i] for i in rand_idx],siamese_targets
        
        else:
            graphs.pop(0)
            bgs=list()
            for i in range(0,len(graphs),self.N):
                inter_gs=[self._assign_node_feat(self._knn_graph(graphs[j],self.knn),tag_atom_resindex[tags[j]],tags[0],self.external_feats) for j in range(i,i+self.N)]
                print('ii1')
                pro_gs=[self._assign_node_feat(self._knn_graph(graphs[j],self.knn,'protein'),tag_atom_resindex[tags[j]],tags[0],self.external_feats) for j in range(i,i+self.N)]
                print('ii2')
                rna_gs=[self._assign_node_feat(self._knn_graph(graphs[j],self.knn,'rna'),tag_atom_resindex[tags[j]],tags[0],self.external_feats) for j in range(i,i+self.N)]
                print('ii3')
                bgs.append((dgl.batch(inter_gs),dgl.batch(pro_gs),dgl.batch(rna_gs)))
            
            with open(os.path.join(self.temp_dataset_dir,tags[0]+'.pkl'),'wb') as f:
                pickle.dump((bgs,targets,tags),f)
            
            return bgs,targets,tags

class HeteroGraphDataset_resgraph_bind(HeteroGraphDataset_resgraph):
    def __init__(self,**kwargs):
        super(HeteroGraphDataset_resgraph_bind,self).__init__(**kwargs)

    def _get_bind_res(self,pos_g,atom_res_index):
        atom_num=pos_g.ndata['atom_number_ori'].tolist()
        bind_res=[atom_res_index[x] for x in atom_num]
        return bind_res
    
    def _get_bind_targets(self,g,atom_res_index,bind_res):
        atom_num=g.ndata['atom_number_ori'].tolist()
        targets=[1 if atom_res_index[x] in bind_res else 0 
            for x in atom_num]
        return torch.tensor(targets).reshape(-1,1)
    
    def __getitem__(self, idx):
        tags,graphs,targets,tag_atom_resindex=self._load_graph([self.hetero_graphs[idx]])[0]
        
        if self.train:
            sample_weights=self._get_rmsds(tags, self.rmsd)
            rand_idx=random.choices(range(1,len(tags)),weights=sample_weights,k=self.N) #exist same value
            rand_idx.insert(0,0)
            
            tags=[tags[i] for i in rand_idx]
            knn_gs=[self._knn_graph(graphs[i],self.knn) for i in rand_idx]
            
            bind_res=self._get_bind_res(knn_gs[0],tag_atom_resindex[tags[0]])
            targets=[self._get_bind_targets(x,tag_atom_resindex[y],bind_res) for x,y in zip(knn_gs,tags)]
            
            gs=[self._assign_node_feat(x,tag_atom_resindex[y],tags[0],self.external_feats) for x,y in zip(knn_gs,tags)]
            return dgl.batch(gs),torch.row_stack(targets),tags
        
        else:
            if os.path.exists(os.path.join(self.temp_dataset_dir,tags[0]+'.pkl')):
                with open(os.path.join(self.temp_dataset_dir,tags[0]+'.pkl'),'rb') as f:
                    bgs,targets,tags=pickle.load(f)
                    print(tags[0])
            else:
                graphs.pop(0)
                bgs=list()
                for i in range(0,len(graphs),self.N):
                    gs=[self._assign_node_feat(self._knn_graph(graphs[j],self.knn),tag_atom_resindex[tags[j]],tags[0],self.external_feats) for j in range(i,i+self.N)]
                    bgs.append(dgl.batch(gs))
                
                with open(os.path.join(self.temp_dataset_dir,tags[0]+'.pkl'),'wb') as f:
                    pickle.dump((bgs,targets,tags),f)
            
            return bgs,targets,tags


# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def success(predicts,tags,rmsddict,topn=[5,10,15,20]):
    predicts=predicts.reshape(-1)
    k=tags.pop(0)
    rmsds=np.array([rmsddict[k][x] for x in tags],dtype=float)
    sorts=np.argsort(predicts)[::-1]
    return np.array([(rmsds[sorts[:x]]<=4).any() for x in topn],dtype=int)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     dgl.seed(seed)
     dgl.random.seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic=True

    