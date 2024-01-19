import os
import re
import io
import sys
import shutil
import numpy as np
import pickle
import math
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import subprocess
import tempfile
from datapre import one_of_k_encoding_unk,run_naccess,get_atom_chain_resindex
import pymol
from PDBfuc import PDB, MPDB
import esm
import fm
import json

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'config.json'),'r') as f:
    para=json.load(f)

# =============================================================================
# # residue one-hot
# =============================================================================
STD_RESIDUE_NAME=[
    "A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","X"
]
STD_NUCLEOTIDE_NAME=[
    "A","G","C","U","X"
]

def fasta_to_onehot(fasta_file,value_field):
    fasta=SeqIO.parse(fasta_file,format='fasta')
    onehot_dict=dict()
    for record in fasta:
        onehot=np.array([one_of_k_encoding_unk(x,value_field) for x in record.seq],dtype=int)
        onehot_dict[record.id]=onehot
    return onehot_dict

def onehot(fasta_pro,fasta_rna,savepath):
    feat_dict=dict()
    feat_dict['test']={**fasta_to_onehot(fasta_pro,STD_RESIDUE_NAME),
                       **fasta_to_onehot(fasta_rna,STD_NUCLEOTIDE_NAME)}
    with open(savepath,'wb') as f:
        pickle.dump(feat_dict,f)


# =============================================================================
# # residue position
# =============================================================================
def fasta_to_position(fasta_file):
    fasta=SeqIO.parse(fasta_file,format='fasta')
    position_dict=dict()
    for record in fasta:
        position=np.array([i/len(record.seq) for i in range(len(record.seq))]).reshape(-1,1)
        position_dict[record.id]=position
    return position_dict

def position(fasta_pro,fasta_rna,savepath):
    feat_dict=dict()
    feat_dict['test']={**fasta_to_position(fasta_pro),
                       **fasta_to_position(fasta_rna)}
    with open(savepath,'wb') as f:
        pickle.dump(feat_dict,f)


# =============================================================================
# # residue secondary structure
# =============================================================================
STD_RESIDUE_SS=['H','E','C']
RESIDUE_SS_MAP={'H':'H','G':'H','I':'H',
                'E':'E','B':'E',
                'T':'C','S':'C','L':'C','-':'C'}

def get_ss_protein(protein):
    temp_dir=tempfile.mkdtemp()
    with open(os.path.join(temp_dir,'protein.pdb'),'w') as f:
        f.write(protein)
    
    p=PDBParser()
    structure=p.get_structure('protein',os.path.join(temp_dir,'protein.pdb'))
    dssp=DSSP(structure[0],os.path.join(temp_dir,'protein.pdb'),
              dssp=para['dssp'])
    
    sslist=list()
    for res in structure.get_residues():
        if res.full_id[2:] in dssp:
            sslist.append(dssp[res.full_id[2:]][2])
        else:
            sslist.append('-')
    
    ss_onehot=np.array([one_of_k_encoding_unk(RESIDUE_SS_MAP[x], STD_RESIDUE_SS) for x in sslist],dtype=int)
    shutil.rmtree(temp_dir)
    return ss_onehot

def get_ss_rna(rna):
    def count(typestr):#eg. anti,~C3'-endo,BI,canonical,non-pair-contact,helix,stem,coaxial-stack
          judgelist=['bulge','ss-non-loop','stem','internal-loop','hairpin-loop','junction-loop','pseudoknotted']
          return list(map(lambda x:1 if re.search(x,typestr) else 0,judgelist))
    
    temp_dir=tempfile.mkdtemp()
    os.chdir(temp_dir)
    with open('rna.pdb','w') as f:
        f.write(rna)
    
    pdb=PDB('rna.pdb')
    res_num=[x[2] for x in pdb.xulie]
    
    dssr_command=[
                  para['dssr'],
                  '-i=rna.pdb',
                  '-o={}'.format(os.path.join(temp_dir,'output.txt'))
                  ]
    subprocess.run(dssr_command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
    
    with open(os.path.join(temp_dir,'output.txt'),'r') as f:
        output=f.readlines()
    ss_field=list(filter(lambda x:re.match('\s+\d+\s+\w\s+.\s+\w\.',x),output))
    
    ss_descriptors=list()
    if len(ss_field)<len(res_num):
        print('Warning: length ss_field is {} while res_len is {}'.format(len(ss_field),len(res_num)))
        for i in range(len(res_num)):
            res=res_num[i]
            res_len=len(res)
            if i<len(res_num)-1:
                l_res=ss_field[i].split()[3].replace('^','')[-res_len:]
            elif i==len(res_num)-1 and i==len(ss_field)-1:
                l_res=ss_field[i].split()[3].replace('^','')[-res_len:]
            else:
                ss_descriptors.append('None')
                break
            
            if l_res!=res:
                print('Warning: add res {} at {}'.format(res,i))
                ss_descriptors.append('None')
                ss_field.insert(i,None)
            else:
                ss_descriptors.append(ss_field[i].split()[5])
    
    elif len(ss_field)==len(res_num):
        ss_descriptors=[x.split()[5] for x in ss_field]
    else:
        raise ValueError('ERROR for dssr_ss_field')
        
    ss_onehot=np.row_stack([count(x) for x in ss_descriptors]).astype(int)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    shutil.rmtree(temp_dir)
    return ss_onehot

def get_protein_rna(pymol_pdb):
    pdb_id=os.path.basename(pymol_pdb).replace('_fix_pymol_4.5A.pkl','')
    with open(pymol_pdb,'rb') as f:
        pdb_dict,chain_atom_idt,interface_atom_idt,molecular_atom_idt=pickle.load(f)
    
    monomer_dict=dict()
    for chain,atom_idt in chain_atom_idt.items():
        if any([x in molecular_atom_idt['protein'] for x in atom_idt]):
            k='{}_protein_{}'.format(pdb_id,chain)
        elif any([x in molecular_atom_idt['rna'] for x in atom_idt]):
            k='{}_rna_{}'.format(pdb_id,chain)
        else:
            raise ValueError('atom_idt ERROR')
        
        monomer_dict[k]='\n'.join([pdb_dict['dict_pdb'][x] for x in atom_idt])
    return monomer_dict

def get_ss_feat_for_dir(pymol_pdb):
    dir_feat_dict=dict()
    for x in pymol_pdb:
        print(x,end='\t')
        monomer_dict=get_protein_rna(x)
        for k,v in monomer_dict.items():
            print(k,end='\t')
            if re.search('protein',k):
                dir_feat_dict[k]=get_ss_protein(v)
            elif re.search('rna',k):
                dir_feat_dict[k]=get_ss_rna(v)
            else:
                raise ValueError('monomer ERROR')
        print()
    return dir_feat_dict

def ss(pymol_pdb,savepath):
    feat_dict=dict()
    feat_dict['test']={**get_ss_feat_for_dir([pymol_pdb])}
    with open(savepath,'wb') as f:
        pickle.dump(feat_dict,f)


# =============================================================================
# # residue backbone torsion angle
# =============================================================================
def sin_cos_trans(angle):
    sin=math.sin(math.radians(angle))
    cos=math.cos(math.radians(angle))
    return sin,cos

def get_angle_protein(protein):
    temp_dir=tempfile.mkdtemp()
    with open(os.path.join(temp_dir,'protein.pdb'),'w') as f:
        f.write(protein)
    
    p=PDBParser()
    structure=p.get_structure('protein',os.path.join(temp_dir,'protein.pdb'))
    dssp=DSSP(structure[0],os.path.join(temp_dir,'protein.pdb'),
              dssp=para['dssp'])
    
    philist=list()
    psilist=list()
    for res in structure.get_residues():
        if res.full_id[2:] in dssp:
            philist.append(sin_cos_trans(dssp[res.full_id[2:]][4]))
            psilist.append(sin_cos_trans(dssp[res.full_id[2:]][5]))
        else:
            print('Warning! add zero for phi and psi')
            philist.append(sin_cos_trans(0.))
            psilist.append(sin_cos_trans(0.))
    
    shutil.rmtree(temp_dir)
    return np.hstack((np.array(philist),np.array(psilist)))

def split_pdb(pdb):
    pymol.cmd.delete('all')
    pymol.cmd.load(pdb)
    pymol.cmd.select('pro','polymer.protein')
    pymol.cmd.select('rna','polymer.nucleic')
    pymol.cmd.save(pdb.replace('.pdb','_protein.pdb'),'pro')
    pymol.cmd.save(pdb.replace('.pdb','_rna.pdb'),'rna')

def get_angle_rna(rna):
    def float_angle(angle):
        if angle=='---':
            return 0.
        else:
            return float(angle)
    
    temp_dir=tempfile.mkdtemp()
    os.chdir(temp_dir)
    with open('rna.pdb','w') as f:
        f.write(rna)
    
    pdb=PDB('rna.pdb')
    res_num=[x[2] for x in pdb.xulie]
    
    dssr_command=[
                  para['dssr'],
                  '-i=rna.pdb',
                  '-o={}'.format(os.path.join(temp_dir,'output.txt'))
                  ]
    subprocess.run(dssr_command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
    
    with open(os.path.join(temp_dir,'dssr-torsions.txt'),'r') as f:
        output=f.readlines()
    torsion_line_start=list(filter(lambda x:re.match('''\s+nt\s+eta\s+theta\s+eta'\s+theta'\s+eta"\s+theta"''',x),output))
    assert len(torsion_line_start)==1
    start_index=output.index(torsion_line_start[0])+1
    
    etalist=list()
    thetalist=list()
    eta1list=list()
    theta1list=list()
    eta2list=list()
    theta2list=list()
    l_res_list=list()
    for line in output[start_index:]:
        if re.match('\*\*\*\*',line):
            break
        
        linelist=line.split()
        l_res_list.append(linelist[2])
        
        etalist.append(sin_cos_trans(float_angle(linelist[3])))
        thetalist.append(sin_cos_trans(float_angle(linelist[4])))
        eta1list.append(sin_cos_trans(float_angle(linelist[5])))
        theta1list.append(sin_cos_trans(float_angle(linelist[6])))
        eta2list.append(sin_cos_trans(float_angle(linelist[7])))
        theta2list.append(sin_cos_trans(float_angle(linelist[8])))
    
    if len(l_res_list)<len(res_num):
        print('Warning: length torsion is {} while res_len is {}'.format(len(l_res_list),len(res_num)))
        for i in range(len(res_num)):
            res=res_num[i]
            res_len=len(res)
            
            if i==len(res_num)-1:
                if i!=len(l_res_list)-1:
                    print('Warning: add finally one')
                    etalist.append((0,1))
                    thetalist.append((0,1))
                    eta1list.append((0,1))
                    theta1list.append((0,1))
                    eta2list.append((0,1))
                    theta2list.append((0,1))
                    break
            
            if l_res_list[i].replace('^','')[-res_len:]!=res_num[i]:
                print('Warning: add res{} at {}'.format(res_num[i],i))
                l_res_list.insert(i,None)
                etalist.insert(i,(0,1))
                thetalist.insert(i,(0,1))
                eta1list.insert(i,(0,1))
                theta1list.insert(i,(0,1))
                eta2list.insert(i,(0,1))
                theta2list.insert(i,(0,1))
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    shutil.rmtree(temp_dir)
    return np.hstack((etalist,thetalist,eta1list,theta1list,eta2list,theta2list))

def get_universal_feat_for_dir(pymol_pdb,protein_fuc,rna_fuc):
    dir_feat_dict=dict()
    for x in pymol_pdb:
        print(x,end='\t')
        monomer_dict=get_protein_rna(x)
        for k,v in monomer_dict.items():
            print(k,end='\t')
            if re.search('protein',k):
                dir_feat_dict[k]=protein_fuc(v)
            elif re.search('rna',k):
                dir_feat_dict[k]=rna_fuc(v)
            else:
                raise ValueError('monomer ERROR')
        print()
    return dir_feat_dict

def get_fasta(pdb,types):
    pdbname=os.path.basename(pdb).replace('.pdb','')
    mpdb=MPDB(pdb)
    fasta=dict()
    for restype,chain,res in mpdb.xulie:
        header='>{}_{}'.format(pdbname,chain)
        if header not in fasta:
            fasta.setdefault(header,[])
        if types=='protein':
            fasta[header].append(mpdb.three_to_one[mpdb.het_to_atom[restype]])
        else:
            if restype=='RU':
                fasta[header].append('U')
            else:
                fasta[header].append(mpdb.het_to_atom[restype])
    for k in fasta:
        assert all(len(x)==1 for x in fasta[k])
    return fasta

def fasta_all(pdb):
    split_pdb(pdb)
    fasta_pro=get_fasta(pdb.replace('.pdb','_protein.pdb'), types='protein')
    fasta_rna=get_fasta(pdb.replace('.pdb','_rna.pdb'), types='rna')
    with open(os.path.dirname(pdb)+'/'+'all_protein.fasta','w') as f:
        for k,v in fasta_pro.items():
            f.write(k+'\n')
            f.write(''.join(v)+'\n')
    with open(os.path.dirname(pdb)+'/'+'all_rna.fasta','w') as f:
        for k,v in fasta_rna.items():
            f.write(k+'\n')
            f.write(''.join(v)+'\n')

def torsion(pymol_pdb,savepath):
    feat_dict=dict()
    feat_dict['test']={**get_universal_feat_for_dir([pymol_pdb],get_angle_protein,get_angle_rna)}
    with open(savepath,'wb') as f:
        pickle.dump(feat_dict,f)


# =============================================================================
# # residue RSA or LAP
# =============================================================================
from FEAfuc import RSA,Laplacian,Topo
from GeometricalFeatfuc import MathMorphologyPocket

def get_rsa(monomer):
    temp_dir=tempfile.mkdtemp()
    with open(os.path.join(temp_dir,'monomer.pdb'),'w') as f:
        f.write(monomer)
    
    p=PDBParser()
    structure=p.get_structure('monomer',os.path.join(temp_dir,'monomer.pdb'))
    _,asa_data=run_naccess(model=structure[0],pdb_file=None,probe_size=1.5,hetatm=True)
    with open(os.path.join(temp_dir,'monomer.asa'),'w') as f:
        f.write(''.join(asa_data))
        
    rsa=RSA(asafile=os.path.join(temp_dir,'monomer.asa'),pdbfile=os.path.join(temp_dir,'monomer.pdb')).res_array().reshape(-1,1)
    shutil.rmtree(temp_dir)
    return rsa

def get_lap(monomer):
    temp_dir=tempfile.mkdtemp()
    with open(os.path.join(temp_dir,'monomer.pdb'),'w') as f:
        f.write(monomer)
    
    lap=Laplacian(os.path.join(temp_dir,'monomer.pdb')).res_array('average')
    lap=np.nan_to_num(lap)
    
    shutil.rmtree(temp_dir)
    return lap

def get_pocketness(monomer):
    temp_dir=tempfile.mkdtemp()
    with open(os.path.join(temp_dir,'monomer.pdb'),'w') as f:
        f.write(monomer)
    
    #/work/home/lr_hzau/jz/soft/ghecom/ghecom
    os.system('{} -M M -atmhet B -hetpep2atm F -ipdb {} -ores {}'.format(para['ghecom'],os.path.join(temp_dir,'monomer.pdb'),os.path.join(temp_dir,'monomer.out')))
    pocketness=MathMorphologyPocket().res_array(os.path.join(temp_dir,'monomer.out'))
    
    pocketness=pocketness[:,1:].astype(float)
    col1=pocketness[:,0]/100
    col2=(pocketness[:,1].clip(None,20))/10
    col3=pocketness[:,2]/100
    
    shutil.rmtree(temp_dir)
    return np.column_stack((col1,col2,col3))

def get_topo_pro(monomer):
    temp_dir=tempfile.mkdtemp()
    with open(os.path.join(temp_dir,'monomer.pdb'),'w') as f:
        f.write(monomer)
    
    topo=Topo(os.path.join(temp_dir,'monomer.pdb')).res_array(5)
    shutil.rmtree(temp_dir)
    return topo

def get_topo_rna(monomer):
    temp_dir=tempfile.mkdtemp()
    with open(os.path.join(temp_dir,'monomer.pdb'),'w') as f:
        f.write(monomer)
    
    topo=Topo(os.path.join(temp_dir,'monomer.pdb')).res_array(8)
    shutil.rmtree(temp_dir)
    return topo

def get_usr(monomer):
    temp_dir=tempfile.mkdtemp()
    with open(os.path.join(temp_dir,'monomer.pdb'),'w') as f:
        f.write(monomer)
    
    usr=UltrafastShape(os.path.join(temp_dir,'monomer.pdb')).res_array()
    shutil.rmtree(temp_dir)
    return usr

def rsa(pymol_pdb,savepath):
    feat_dict=dict()
    feat_dict['test']={**get_universal_feat_for_dir([pymol_pdb],get_rsa,get_rsa)}
    with open(savepath,'wb') as f:
        pickle.dump(feat_dict,f)

def lap(pymol_pdb,savepath):
    feat_dict=dict()
    feat_dict['test']={**get_universal_feat_for_dir([pymol_pdb],get_lap,get_lap)}
    with open(savepath,'wb') as f:
        pickle.dump(feat_dict,f)

def pocketness(pymol_pdb,savepath):
    feat_dict=dict()
    feat_dict['test']={**get_universal_feat_for_dir([pymol_pdb],get_pocketness,get_pocketness)}
    with open(savepath,'wb') as f:
        pickle.dump(feat_dict,f)

def ResidueEmbedding(fasta_pro,savepath):
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(para['checkpoint_for_esm'])
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    model=model.cuda()
    with open(fasta_pro,'r') as f:
        flist=f.readlines()
    out=dict()
    for i in range(0,len(flist),2):
        print(i)
        data=[(flist[i][1:].strip(),flist[i+1].strip())]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens=batch_tokens.cuda()
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33][0][1:-1,:].cpu().numpy()
        # torch.save(results,'./data/dna/feature/esm2/'+data[0][0]+'.pt')
        out[data[0][0]]=token_representations
    with open(savepath,'wb') as f:
        pickle.dump(out,f)

def NucleotideEmbedding(fasta_rna,savepath):
    model, alphabet = fm.pretrained.rna_fm_t12(para['checkpoint_for_fm'])
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    fasta=np.loadtxt(fasta_rna,dtype=str)
    embed=dict()
    for i in range(0,len(fasta),2):
        print(i)
        data=[(str(fasta[i][1:].strip()),str(fasta[i+1].strip()))]
    
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        # Extract embeddings (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12], need_head_weights=True, return_contacts=True)
        
        embed[data[0][0]]=results["representations"][12].cpu().numpy()
        
        # with open('./data/ligand/feature/FM/{}.pkl'.format(data[0][0]),'wb') as f:
        #     pickle.dump(results,f)
    
    with open(savepath,'wb') as f:
        pickle.dump(embed,f)

