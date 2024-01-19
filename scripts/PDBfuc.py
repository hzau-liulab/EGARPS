import re
import os
import numpy as np
import itertools
import pandas as pd
from collections import Counter
import string

import pickle

import pymol


class PDB(object):
    """
    NOTE: only single chain of pdb is accepted
    het_to_atom => dict contains het res to res and DNA res to one letter res (eg. DT => T)
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'hetatm.pkl'),'rb') as f:
        het_to_atom=pickle.load(f)
    
    prorestypelist=['ALA','ARG','ASN','ASP','CYS',
                    'GLN','GLU','GLY','HIS','ILE',
                    'LEU','LYS','MET','PHE','PRO',
                    'SER','THR','TRP','TYR','VAL',]
    
    three_to_one={'GLY':'G','ALA':'A','VAL':'V','LEU':'L','ILE':'I',
                  'PHE':'F','TRP':'W','TYR':'Y','ASP':'D','ASN':'N',
                  'GLU':'E','LYS':'K','GLN':'Q','MET':'M','SER':'S',
                  'THR':'T','CYS':'C','PRO':'P','HIS':'H','ARG':'R',
                  'UNK':'X',}
    
    letter=string.ascii_uppercase
    
    def __init__(self,pdbfile,autofix=True,keephetatm=True,keepalternativeres=True,autocheck=False):
        """
        keephetatm: bool
        keepalternativeres: bool
        autocheck: bool (whether to remove residue without enough atoms)
        self.res_atom: list => [[res,atomtype],......]
        self.coord: dict => {res:{atomtype:[x,y,z]},.....}
        self.coord_resatom: dict => {(x,y,z):[res,atomtype],.....}
        """
        self.xulie=list()
        self.res_atom=list()
        self.coord=dict()
        self.coord_resatom=dict()
        
        if isinstance(pdbfile,(np.ndarray,list)):
            self.pdb=pdbfile
        elif isinstance(pdbfile,str):
            self.pdb=open(pdbfile).readlines()
        elif pdbfile==None:
            return 
        self._pdb_deal(keephetatm,keepalternativeres,autocheck)
        
        self._extraction()
        # self._xulie_deal()
        if len(self.xulie)!=len(self.coord) and autofix:
            self._pdb_res_fix()
            self.xulie=list()
            self.res_atom=list()
            self.coord=dict()
            self.coord_resatom=dict()
            self._extraction()
        
        self.moleculer_type=self._check_type()
        self.het_to_atom=self.het_to_atom[self.moleculer_type]
        
        self.index_res={i:j[-1] for i,j in enumerate(self.xulie)}
        self.res_index={j[-1]:i for i,j in enumerate(self.xulie)}
        
        pass
    
    def _xulie(self,pdb_line):
        res_type=pdb_line[17:20].strip()
        chain=pdb_line[21:22].strip()
        res=pdb_line[22:28].strip()
        return res_type,chain,res
    
    # def _xulie_deal(self):
    #     """
    #     eg. change [['A','C','60'],['G','C','60']] to [['A','C','60A'],['G','C','60B']]
    #     """
    #     reslist=[x[-1] for x in self.xulie]
    #     def fuc(x):
    #         res,count=x
    #         index=list(filter(lambda k:reslist[k]==res,range(len(reslist))))
    #         for i,j in zip(index,range(count)):
    #             self.xulie[i][-1]=reslist[i]+self.letter[j]
        
    #     counter=Counter(reslist)
    #     element=list(filter(lambda x:x[1]>1,counter.most_common()))
    #     if element:
    #         for x in element:
    #             fuc(x)
    
    def _coord(self,pdb_line):
        x=float(pdb_line[30:38].strip())
        y=float(pdb_line[38:46].strip())
        z=float(pdb_line[46:54].strip())
        return x,y,z
    
    def _atom(self,pdb_line):
        """
        Alternative also included, such as C3'A and C3'B
        """
        return pdb_line[12:17].strip()
    
    def _mix_fuc(self,pdb_line):
        return self._xulie(pdb_line)+self._coord(pdb_line)+(self._atom(pdb_line),)
    
    def _extraction(self):
        tmp_tuple=map(lambda x:self._mix_fuc(x),self.pdb)
        tmp_dict=dict()
        
        for x in tmp_tuple:
            if [x[0],x[1],x[2]] not in self.xulie:
                self.xulie.append([x[0],x[1],x[2]])
            tmp_dict.setdefault(x[2],[]).append((x[6],x[3:6]))
            self.coord_resatom[tuple(x[3:6])]=[x[2],x[6]]
            self.res_atom.append((x[2],x[6]))
        
        for key,value in tmp_dict.items():
            self.coord[key]=dict(value)
    
    def _extraction2(self):
        tmp_tuple=map(lambda x:self._mix_fuc(x),self.pdb)
        tmp_dict=dict()
        
        for x in tmp_tuple:
            if [x[0],x[1],x[2]+x[0]] not in self.xulie:
                self.xulie.append([x[0],x[1],x[2]+x[0]])
            tmp_dict.setdefault(x[2]+x[0],[]).append((x[6],x[3:6]))
            self.coord_resatom[tuple(x[3:6])]=[x[2]+x[0],x[6]]
            self.res_atom.append((x[2]+x[0],x[6]))
        
        for key,value in tmp_dict.items():
            self.coord[key]=dict(value)
    
    def res_distance(self,mode='min'):
        """
        mode can be min, average, or atom name (eg. C3')
        min: the min distance
        average: the distance calcaluted by average coord
        atom: the distance calcaluted by atom coord
        return dict => {res1:(res2,distance),.......}
        """
        res_coord=self._coord_deal(mode)
        
        dim=len(self.index_res)
        index=np.where(np.triu(np.ones(shape=(dim,dim)),k=1))
        
        res_dist=dict()
        for i,j in zip(index[0],index[1]):
            res_i=self.index_res[i]
            res_j=self.index_res[j]
            coord_i=res_coord[res_i]
            coord_j=res_coord[res_j]
            dist=list(map(lambda x:self.cal_distance(x[0],x[1]),
                     itertools.product(coord_i,coord_j)))
            distance=min(dist)
            res_dist.setdefault(res_i,[]).append((res_j,distance))
            res_dist.setdefault(res_j,[]).append((res_i,distance))
        
        for _,res in self.index_res.items():
            res_dist.setdefault(res,[]).append((res,0.))
        
        return res_dist
        
    def cal_distance(self,vec1,vec2):
        vec1=np.array(vec1,dtype=float)
        vec2=np.array(vec2,dtype=float)
        return np.sqrt(np.sum(np.square(vec1-vec2)))
    
    def cal_angle(self,vec1,vec2):
        vec1=np.array(vec1,dtype=float)
        vec2=np.array(vec2,dtype=float)
        vec1=vec1/np.linalg.norm(vec1)
        vec2=vec2/np.linalg.norm(vec2)
        return np.arccos(np.clip(np.dot(vec1,vec2),-1.,1.))
    
    def _coord_deal(self,mode='average'):
        """
        mode can be min, average, or atom name
        return dict => {res1:[(x1,y1,z1),(x2,y2,z2)]}
        """
        # TODO this function is too reduency with so many for cell, optim is needed
        
        res_coord=dict.fromkeys(self.coord.keys())
        
        if mode=='min':
            for res in self.coord:
                res_coord[res]=list(map(lambda x:list(self.coord[res][x]),
                                        self.coord[res]))
            return res_coord
        
        elif mode=='average':
            for res in self.coord:
                average_coord=[]
                for atom,value in self.coord[res].items():
                    average_coord.append(list(value))
                average_coord=np.array(average_coord)
                res_coord[res]=[np.mean(average_coord,axis=0).tolist()]
            return res_coord
        
        else:
            # TODO in the statr or end of the pdb, there maybe some atom miss
            # what do we handle this
            # (for now, there is an error occure for this condition)
            for res in self.coord:
                if mode in self.coord[res]:
                    res_coord[res]=[list(self.coord[res][mode])]
                elif any(re.match(mode,x) for x in self.coord[res]):
                    average_coord=[]
                    for atom,value in self.coord[res].items():
                        if re.match(mode,atom):
                            average_coord.append(list(value))
                    average_coord=np.array(average_coord)
                    res_coord[res]=[np.mean(average_coord,axis=0).tolist()]
                else:
                    raise ValueError('check the input atom name (mode para)')
            return res_coord
    
    def _pdb_deal(self,keephetatm=True,keepalternativeres=True,autocheck=False):
        """
        remove atom H
        """
        self.pdb=list(filter(lambda x:any((re.match('ATOM',x),re.match('HETATM',x))) \
                             and x.strip().split()[-1] != 'H',
                             self.pdb))
        if keephetatm==False:
            self.pdb=self.remove_hetatm()
        if keepalternativeres==False:
            self.pdb=self.remove_alternative()
        if autocheck==True:
            self.pdb=self.correction()
    
    def _pdb_res_fix(self):
        """
        same res to add different letter eg. 60,60 => 60A,60B
        """
        reslist=[x[-1] for x in self.xulie]
        counter=Counter(reslist)
        element=list(filter(lambda x:x[1]>1,counter.most_common()))
        for res,count in element:
            tmpindex=list(filter(lambda x:self.pdb[x][22:28].strip()==res,range(len(self.pdb))))
            tmpres_type=[self.pdb[i][17:20].strip() for i in tmpindex]
            tmpres_type2=sorted(list(set(tmpres_type)),key=lambda x:tmpres_type.index(x))
            tmpdict=dict(zip(tmpres_type2,[' ']+list(self.letter[:len(tmpres_type2)-1])))
            if len(tmpdict)!=count:
                raise ValueError
            for i in tmpindex:
                pdbline=self.pdb[i]
                add=tmpdict[self.pdb[i][17:20].strip()]
                self.pdb[i]=pdbline[:22]+' '*(4-len(res))+res+add+pdbline[27:]
    
    def remove_hetatm(self):
        """
        keep ATOM records only
        """
        return list(filter(lambda x:re.match('ATOM',x),self.pdb))
    
    def remove_alternative(self):
        """
        remove alternative residue eg. 12A
        """
        return list(filter(lambda x:re.search('\d$',x[22:28].strip()),self.pdb))
    
    def correction(self,num=1):
        """
        num: int
        if number of atoms in the res <= num, discard this res
        """
        self.pdb=list(self.pdb)
        rescount=dict()
        for i in range(len(self.pdb)):
            rescount.setdefault(self.pdb[i][22:28].strip(),[])
            rescount[self.pdb[i][22:28].strip()].append(i)
        delres=list(filter(lambda x:len(rescount[x])<=num,rescount))
        delindex=sum([rescount[x] for x in delres],[])
        index=[x for x in range(len(self.pdb)) if x not in delindex]
        return [self.pdb[x] for x in index]
    
    def distance_map(self,mode='average'):
        """
        mode can be min, average, or atom name
        return array of float
        """
        dim=len(self.res_index)
        self.distmap=np.zeros((dim,dim))
        for key,value in self.res_distance(mode).items():
            index_i=self.res_index[key]
            for x,dist in value:
                index_j=self.res_index[x]
                self.distmap[index_i,index_j]=dist
        return self.distmap
    
    def contact_map(self,distcut=8,mode='average'):
        """
        mode can be min, average, or atom name
        return array of int
        self contact is not included
        """
        distmap=self.distance_map(mode)
        contmap=np.ones(distmap.shape,dtype=np.int)
        contmap[np.where(distmap>distcut)]=0
        contmap[np.diag_indices_from(contmap)]=0
        return contmap
    
    def angle_map(self,threeatoms=None):
        """
        threeatoms: list of atoms to decide the panel
        default ['CA','C','N'] for protein ["C3'","C1'","C5'"] for NucleicAcid
        return array of float
        """
        # if threeatoms is None:
        #     threeatoms=['CA','C','N'] if self.moleculer_type=='Protein' else ["C3'","C1'","C5'"]
        
        def get_surface_normal(res):
            threeatoms=[["C3'","C1'","C5'"],['CA','C','N']] 
            if all(x in self.coord[res] for x in threeatoms[0]): #assume it is NA
                vectors=np.array([self.coord[res][x] for x in threeatoms[0]],dtype=float)
                normal=np.cross(vectors[1,:]-vectors[0,:],vectors[2,:]-vectors[0,:])
            elif all(x in self.coord[res] for x in threeatoms[1]): #assume it is protein
                vectors=np.array([self.coord[res][x] for x in threeatoms[1]],dtype=float)
                normal=np.cross(vectors[1,:]-vectors[0,:],vectors[2,:]-vectors[0,:])
            elif len(self.coord[res])==1:
                normal=list(self.coord[res].values())[0]
            elif len(self.coord[res])>1:
                normal=np.mean(np.array(list(self.coord[res].values()),dtype=float),axis=0)
            else:
                raise ValueError(res)
            normal=np.array(normal,dtype=float)
            return normal
        
        ress=[x[-1] for x in self.xulie]
        anglemap=np.zeros((len(ress),len(ress)))
        for res1,res2 in itertools.product(ress,ress):
            i=self.res_index[res1]
            j=self.res_index[res2]
            if i==j:
                continue
            n1=get_surface_normal(res1)
            n2=get_surface_normal(res2)
            anglemap[i,j]=self.cal_angle(n1,n2)
        return anglemap
        
    def get_res_coord_dict(self,mode='average'):
        """
        mode can be average or atom name
        return dict => {res:[x,y,z],res:[x,y,z],.....}
        """
        res_coord=dict(map(lambda x:(x[0],np.squeeze(x[1]).tolist()),
                           self._coord_deal(mode).items()))
        return res_coord
    
    def get_res_coord_list(self,mode='average'):
        res_coord=self.get_res_coord_dict(mode=mode)
        return np.array([res_coord[res] for _,_,res in self.xulie],dtype=float)
    
    def get_atom_coord_list(self):
        """
        return: ndarray of float (NO. of atoms * 3 (x,y,z))
        """
        return np.array(list(map(lambda x:self.coord[x[0]][x[1]],self.res_atom)))
    
    def get_res_seq_windows(self,windowsize,pdbname=None):
        """
        length of output segment is windowsize*2+1
        pdbname => 1egk_A/1egk
        return dict => {res1:[res2,res3......],.....}
        """
        res_seq=['NA',]*windowsize+[x[-1] for x in self.xulie]+['NA',]*windowsize
        res_window=dict(map(lambda x:(res_seq[x],[res_seq[y] for y in range(x-windowsize,x+windowsize+1)]),
                            range(windowsize,len(res_seq)-windowsize)))
        if isinstance(pdbname,str):
            return dict(map(lambda x:(self._full_name_res(x[0],pdbname),self._full_name_res(x[1],pdbname)),
                            res_window.items()))
        return res_window
    
    def get_res_str_windows(self,distcut=10):
        
        pass
    
    def _full_name_res(self,res,pdbname):
        """
        res => res (str/list/array/tuple)
        pdbname => 1egk_A (str)
        return 1egk_A_res (str/list)
        """
        if isinstance(res,str):
            return pdbname+'_'+res
        elif isinstance(res,(list,np.ndarray,tuple)):
            return list(map(lambda x:pdbname+'_'+x,res))
        else:
            raise ValueError('not accepted type in res attr')
    
    def one_hot_seq(self,expand_to=0,mode='NA'):
        """
        expand_to => expand sequence to length (value < length of sequence means no expand)
        mode => NA or protein
        return => index (list) (position of the oriseq in expanded seq), ndarray (int) (one hot)
        """
        oriseqlist=list(map(lambda x:self.het_to_atom[x[0]],self.xulie))
        if int(expand_to)<=len(oriseqlist):
            seqlist=oriseqlist
            index=list(range(len(seqlist)))
        else:
            seqlist=['X',]*expand_to
            index_start=int(int(expand_to)/2-int(len(oriseqlist))/2)
            index_end=int(index_start+len(oriseqlist))
            index=list(range(index_start,index_end))
            seqlist[index_start:index_end]=oriseqlist
        
        tdict={'X':np.array([0,]*4)}
        if mode=='NA':
            tdict.update(dict(zip(['A','G','C','U'],np.eye(4,dtype=np.int))))
            tdict['T']=np.array([0,0,0,1])
            tdict['I']=np.array([0,0,0,0])
        elif mode=='protein':
            tdict.update(dict(zip(self.prorestypelist,np.eye(20,dtype=np.int))))
        else:
            raise ValueError('Not accepted mode: {}'.format(mode))
        
        return index,np.array(list(map(lambda x:tdict[x],seqlist)),dtype=np.int)
    
    def _atom_distance(self):
        """
        calculate distance between all atoms in pdb
        return df => row==col==allatoms(pdblines)
        """
        index=list(map(lambda x:'_'.join(x),self.res_atom))
        array=np.zeros(shape=(len(index),len(index)))
        for x,y in [(t1,t2) for t1 in range(len(index)) for t2 in range(t1+1,len(index))]:
            array[x,y]=self.cal_distance(self.coord[index[x].split('_')[0]][index[x].split('_')[1]],
                                         self.coord[index[y].split('_')[0]][index[y].split('_')[1]])
        array=array.T+array
        df=pd.DataFrame(array,index=index,columns=index)
        return df
    
    def _check_type(self):
        """
        return: Protein/NA
        NOTE: when multiple chains pdb used, and chains are not in same moculer type, may lead error, CAUTIONS
        """
        restypelist=[x[0] for x in self.xulie]
        lenarray=np.array([len(x) for x in restypelist],dtype=int)
        leng=len(self.xulie)/2
        if (lenarray==3).sum()>=leng:
            return 'Protein'
        elif (lenarray<3).sum()>=leng:
            return 'NA'
        else:
            return 'Protein' #assume it is protein
        
    def get_sequence(self,mode='oneletter'):
        def heterror(restype):
            for x in ['A','G','C','U']:
                if x in restype:
                    return x
            return 'A'
        
        if mode=='oneletter':
            tmplist=list()
            for x,_,_ in self.xulie:
                if x in self.het_to_atom:
                    tmplist.append(self.het_to_atom[x])
                else:
                    tmplist.append(heterror(x))
            # tmplist=[self.het_to_atom[x] for x,_,_ in self.xulie]
            seqlist=list()
            for x in tmplist:
                if len(x)==1:
                    seqlist.append(x)
                else:
                    seqlist.append(self.three_to_one[x])
        else:
            seqlist=[x for x,_,_ in self.xulie]
        return seqlist


class MPDB(PDB):
    """
    Multiple chains pdb is accepted
    res in PDB class is represtend as res (eg. 12, 12A)
    while in MPDB class is represtend as chain_res (eg. F_12, F_12A)
    """
    def __init__(self,pdbfile,keephetatm=True,keepalternativeres=True,autocheck=False):
        
        super(MPDB,self).__init__(pdbfile,keephetatm=keephetatm,keepalternativeres=keepalternativeres,autocheck=autocheck)
    
    def correction(self,num=1):
        self.pdb=list(self.pdb)
        rescount=dict()
        for i in range(len(self.pdb)):
            rescount.setdefault('_'.join([self.pdb[i][21:22],self.pdb[i][22:28].strip()]),[]).append(i)
        delres=list(filter(lambda x:len(rescount[x])<=num,rescount))
        delindex=sum([rescount[x] for x in delres],[])
        index=[x for x in range(len(self.pdb)) if x not in delindex]
        return [self.pdb[x] for x in index]
    
    def _extraction(self):
        tmp_tuple=map(lambda x:self._mix_fuc(x),self.pdb)
        tmp_dict=dict()
        
        for x in tmp_tuple:
            if [x[0],x[1],x[1]+'_'+x[2]] not in self.xulie:
                self.xulie.append([x[0],x[1],x[1]+'_'+x[2]])
            tmp_dict.setdefault(x[1]+'_'+x[2],[]).append((x[6],x[3:6]))
            self.coord_resatom[tuple(x[3:6])]=[x[1]+'_'+x[2],x[6]]
            self.res_atom.append((x[1]+'_'+x[2],x[6]))
        
        for key,value in tmp_dict.items():
            self.coord[key]=dict(value)
    
    def _extraction2(self):
        tmp_tuple=map(lambda x:self._mix_fuc(x),self.pdb)
        tmp_dict=dict()
        
        for x in tmp_tuple:
            if [x[0],x[1],x[1]+'_'+x[2]+x[0]] not in self.xulie:
                self.xulie.append([x[0],x[1],x[1]+'_'+x[2]+x[0]])
            tmp_dict.setdefault(x[1]+'_'+x[2]+x[0],[]).append((x[6],x[3:6]))
            self.coord_resatom[tuple(x[3:6])]=[x[1]+'_'+x[2]+x[0],x[6]]
            self.res_atom.append((x[1]+'_'+x[2]+x[0],x[6]))
        
        for key,value in tmp_dict.items():
            self.coord[key]=dict(value)
    
    def _pdb_res_fix(self):
        reslist=[x[-1] for x in self.xulie]
        counter=Counter(reslist)
        element=list(filter(lambda x:x[1]>1,counter.most_common()))
        for res,count in element:
            chain,res=res.split('_')
            tmpindex=list(filter(lambda x:self.pdb[x][22:28].strip()==res and self.pdb[x][21:22].strip()==chain,
                                 range(len(self.pdb))))
            tmpres_type=[self.pdb[i][17:20].strip() for i in tmpindex]
            tmpres_type2=sorted(list(set(tmpres_type)),key=lambda x:tmpres_type.index(x))
            tmpdict=dict(zip(tmpres_type2,[' ']+list(self.letter[:len(tmpres_type2)-1])))
            if len(tmpdict)!=count:
                raise ValueError
            for i in tmpindex:
                pdbline=self.pdb[i]
                add=tmpdict[self.pdb[i][17:20].strip()]
                self.pdb[i]=pdbline[:22]+' '*(4-len(res))+res+add+pdbline[27:]
    
    def _atom_distance(self):
        index=list(map(lambda x:'_'.join(x),self.res_atom))
        array=np.zeros(shape=(len(index),len(index)))
        for x,y in [(t1,t2) for t1 in range(len(index)) for t2 in range(t1+1,len(index))]:
            array[x,y]=self.cal_distance(self.coord[index[x].split('_')[0]+'_'+index[x].split('_')[1]][index[x].split('_')[2]],
                                         self.coord[index[y].split('_')[0]+'_'+index[y].split('_')[1]][index[y].split('_')[2]])
        array=array.T+array
        df=pd.DataFrame(array,index=index,columns=index)
        return df


class SplitChains(PDB):
    
    def __init__(self,pdbfile):
        """
        pdbfile: file (eg. 1ddy.pdb pdb1ddy.ent)
        """
        self.pdbfile=pdbfile
        self.pdbname=self.pdbfile[-8:-4]
    
    def All(self):
        
        pass
    
    def Select(self,outfile,chain=None,moltype=None):
        """
        chain: chain (str)
        outfile: path (str)
        moltype: organic/polymer.protein/polymer.nucleic (str)
        Write chain pdb to outfile
        NOTE: when moltype=None outputs will contain HOH or some other ligands
        """
        if moltype!=None and moltype not in ['polymer','polymer.protein','polymer.nucleic']:
            raise ValueError('not accepted moltype')
        
        outpdbname=self.pdbname+'_'+str(chain)+'.pdb'
        outfile=os.path.join(outfile,outpdbname)
        
        pymol.cmd.delete('all')
        pymol.cmd.load(self.pdbfile)
        if all(x!=None for x in (chain,moltype)):
            pymol.cmd.select('sele','chain '+str(chain)+' and '+moltype)
        elif chain!=None:
            pymol.cmd.select('sele','chain '+str(chain))
        elif moltype!=None:
            pymol.cmd.select('sele',moltype)
        else:
            raise ValueError('chain and moltype have not to all be None')
        pymol.cmd.save(outfile,'sele')
        
        # with open(outfile,'r') as f:
        #     self.pdb=f.readlines()
        # self._pdb_deal()
        # with open(outfile,'w') as f:
        #     for line in self.pdb:
        #         f.write(line)
        
        # self._check_hetatom(outfile)
    
    def _check_hetatom(self,pdbfile):
        """
        to check if the pdb contains hetatom which not in our prepared het_to_atom dict
        """
        with open(pdbfile,'r') as f:
            pdb=f.readlines()
        hetatom=list(filter(lambda x:re.match('HETATM',x),pdb))
        if len(hetatom)>0:
            hets=set(list(map(lambda x:self._xulie(x)[0],hetatom)))
            for het in hets:
                if het not in self.het_to_atom:
                    print(pdbfile)
                    print(het)



if __name__=='__main__':
    mpdb=MPDB(r'D:\deepbindRes_models\GraphBindData\data\rna\PSTPRNA\protein\5voe_H.pdb',)
    # anglemap=pdb.angle_map()
    pass