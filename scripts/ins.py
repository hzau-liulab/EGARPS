import os
import re
import sys
import pickle
import numpy as np
from datapre import fixedpdb_chain_interface_index,get_pdb_moltype,compute_surface_proximities,get_atom_chain_resindex,process_pdb_into_graph
from feature_residue import *
import torch
from dataload import HeteroGraphDataset_resgraph_siamese

def get_pdb_represent(pdb,savepath):
    represent=fixedpdb_chain_interface_index(pdb,4.5)
    with open(savepath,'wb') as f:
        pickle.dump(represent,f)
    return represent

def get_graph_represent(pdb_represent,savepath):
    pdb_dict,chain_atom_idt,interface_atom_idt,molecular_atom_idt=pdb_represent
    interface_pdb,moltype=get_pdb_moltype(pdb_dict,interface_atom_idt,molecular_atom_idt)
    pre_computed_feat={# 'dpx_cx_lap':compute_atomic_dpx_cx_lap(pdb_dict,chain_atom_idt,interface_atom_idt),
                        'proximity':compute_surface_proximities(pdb_dict,chain_atom_idt,interface_atom_idt),
                        'moltype':torch.tensor(moltype,dtype=torch.int),
                        'atom_chain_resindex':get_atom_chain_resindex(pdb_dict,chain_atom_idt)}
    g=process_pdb_into_graph(interface_pdb,
                            atom_selection_type='all_atom',
                            knn=24,
                            idt=None,
                            pre_computed_feat=pre_computed_feat)
    graph={#os.path.basename(savepath).replace('.pkl',''):(g,pre_computed_feat['atom_chain_resindex']),
        'complex20240520':(g,pre_computed_feat['atom_chain_resindex'])}
    if savepath:
        with open(savepath,'wb') as f:
            pickle.dump(graph,f)
    return graph

def get_feature_represent(pdb, academic=True):
    lp,lr = fasta_all(pdb)
    pymol_pdb=pdb.replace('.pdb','_fix_pymol_4.5A.pkl')
    od = onehot(os.path.join(os.path.dirname(pdb),'all_protein.fasta'),
           os.path.join(os.path.dirname(pdb),'all_rna.fasta'),
           os.path.join(os.path.dirname(pdb),'onehot.pkl'))
    
    position(os.path.join(os.path.dirname(pdb),'all_protein.fasta'),
             os.path.join(os.path.dirname(pdb),'all_rna.fasta'),
             os.path.join(os.path.dirname(pdb),'position.pkl'))
    
    td = ss(pymol_pdb,os.path.join(os.path.dirname(pdb),'ss.pkl'), academic)
    if not academic:
        with open(os.path.join(os.path.dirname(pdb),'ss.pkl'), 'wb') as f:
            pickle.dump({'test':{k:td['test'][k] if k in td['test'] else np.zeros((v.shape[0], 7)) for k,v in od['test'].items()}}, f)
    td = torsion(pymol_pdb,os.path.join(os.path.dirname(pdb),'backbone_torsion.pkl'),academic)
    if not academic:
        with open(os.path.join(os.path.dirname(pdb),'backbone_torsion.pkl'),'wb') as f:
            pickle.dump({'test':{k:td['test'][k] if k in td['test'] else np.zeros((v.shape[0], 12)) for k,v in od['test'].items()}}, f)
    if academic:
        rsa(pymol_pdb,os.path.join(os.path.dirname(pdb),'rsa.pkl'))
    else:
        with open(os.path.join(os.path.dirname(pdb),'rsa.pkl'), 'wb') as f:
            pickle.dump({'test':{k:np.zeros((v.shape[0], 1)) for k,v in od['test'].items()}}, f)
    lap(pymol_pdb,os.path.join(os.path.dirname(pdb),'lap.pkl'))
    pocketness(pymol_pdb,os.path.join(os.path.dirname(pdb),'pocketness.pkl'))
    ResidueEmbedding(os.path.join(os.path.dirname(pdb),'all_protein.fasta'),
                     os.path.join(os.path.dirname(pdb),'all_protein_ESM2_1280.pkl'))
    NucleotideEmbedding(os.path.join(os.path.dirname(pdb),'all_rna.fasta'),
                        os.path.join(os.path.dirname(pdb),'all_rna_FM.pkl'))

def DataGenerate(pre_graph):
    dataset=HeteroGraphDataset_resgraph_siamese(hetero_graphs=[pre_graph],
                                                train=False,
                                                N=1,
                                                knn=12,
                                                temp_dataset_dir=os.path.dirname(pre_graph),
                                                external_feats=['embedding','onehot','ss','position','backbone_torsion','rsa','lap','pocketness'],
                                                installing_feat_dir=os.path.dirname(pre_graph)+'/')
    
    with open(os.path.join(os.path.dirname(pre_graph),'all_protein_ESM2_1280.pkl'),'rb') as f:
        pro_emb=pickle.load(f)
    with open(os.path.join(os.path.dirname(pre_graph),'all_rna_FM.pkl'),'rb') as f:
        rna_emb=dict()
        for k,v in pickle.load(f).items():
            rna_emb[k]=v[0]
    dataset.test_seq_embed={**pro_emb,**rna_emb}
    
    for _,_,tag in dataset:
        print('Input graph generated: {}'.format(tag[0]))
    
    return

def get_input_data(pdb, academic=True):
    pdb_represent=get_pdb_represent(pdb,pdb.replace('.pdb','_fix_pymol_4.5A.pkl'))
    get_graph_represent(pdb_represent,pdb.replace('.pdb','.pkl'))
    get_feature_represent(pdb, academic)
    DataGenerate(pdb.replace('.pdb','.pkl'))
    return pdb.replace('.pdb','.pkl')

if __name__=='__main__':
    pdb=sys.argv[1]
    print(pdb)
    get_input_data(pdb)