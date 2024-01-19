from PDBfuc import PDB,MPDB

import numpy as np
import pandas as pd
import re
import math
import networkx as nx
import os
import itertools



class RSA(MPDB):
    
    asadict={'A':400,'G':400,'C':350,'U':350,'T':350,'DA':400,'DG':400,'DC':350,'DT':350,'DU':350,
            'ALA':106,'ARG':248,'ASN':157,'ASP':163,'CYS':135,'GLN':198,'GLU':194,'GLY':84,'HIS':184,'ILE':169,
            'LEU':164,'LYS':205,'MET':188,'PHE':197,'PRO':136,'SER':130,'THR':142,'TRP':227,'TYR':222,'VAL':142,
            'N':400,'I':400,
            'UNK':200}
    
    def __init__(self,asafile=None,rsafile=None,pdbfile=None,prob=1.5):
        self.asafile=asafile
        self.rsafile=rsafile
        self.pdbfile=pdbfile
        if self.asafile==None:
            self.asafile=os.path.join(os.path.dirname(os.path.realpath(self.pdbfile)),os.path.basename(self.pdbfile).split('.')[0]+'.asa')
            if not os.path.exists(self.asafile):
                self.naccess(prob=prob)
        
        super(RSA,self).__init__(pdbfile)
        
        self.res_atom_asa=dict()
        
    def naccess(self,prob=1.5):
        """
        perform naccess algorithm
        """
        os.system('naccess -p '+str(prob)+' -h '+self.pdbfile)
        self.asafile=os.path.join(os.path.dirname(os.path.realpath(self.pdbfile)),os.path.basename(self.pdbfile).split('.')[0]+'.asa')
    
    def residue(self,relative=True):
        """
        residue level asa
        relative: to calculate relative asa
        return dict => {res1:asa,res2:asa,......}
        """
        if not self.res_atom_asa:
            self.atom()
        
        res_asa={}
        for res in self.res_atom_asa:
            res_asa[res]=np.sum(np.vstack(tuple(np.array(x[1]).reshape(1,-1) for x in self.res_atom_asa[res].items())),axis=0)
        
        if relative:
            for key in res_asa:
                res_asa[key]=res_asa[key]/self.asadict[self.het_to_atom[self.xulie[self.res_index[key]][0]]]
        return res_asa
    
    def res_array(self,relative=True):
        """
        return ndarray float
        """
        res_asa=self.residue(relative=relative)
        return np.array([res_asa[x[2]] for x in self.xulie],dtype=float)
    
    def atom(self):
        """
        atom level asa, asafile needed
        return dict => {res1:{atomtype1:asa,atomtype2:,asa},.......}
        """
        res_atom_asa=dict()
        with open(self.asafile,'r') as f:
            for line in f.readlines():
                res_atom_asa.setdefault(line[21:22]+'_'+line[22:28].strip(),{})
                res_atom_asa[line[21:22]+'_'+line[22:28].strip()][line[12:17].strip()]=float(line[54:62].strip())
        
        self.res_atom_asa=self._check(res_atom_asa,mode='atom')
        return self.res_atom_asa
    
    def atom_array(self):
        """
        return: ndarray (number of atoms*1)
        """
        if not self.res_atom_asa:
            self.atom()
        return np.array(list(map(lambda x:self.res_atom_asa[x[0]][x[1]],self.res_atom))).reshape(-1,1)
    
    def _check(self,dictin,mode='atom'):
        """
        to check if the residues(atoms) in asafile is equal to the pdbfile
        if not equal, print the missing info and add value -1 to the missings
        mode => atom or residue
        return dict
        """
        for res in self.coord:
            if res not in dictin:
                if mode=='residue':
                    # print('missing residue {}'.format(res))
                    dictin[res]=-1.
                elif mode=='atom':
                    # print('missing all atoms in residue {}'.format(res))
                    dictin[res]=dict.fromkeys(self.coord[res].keys(),-1.)
            else:
                if mode=='atom':
                    for atom in self.coord[res]:
                        if atom not in dictin[res]:
                            # print('missing atom {} in residue {}'.format(atom,res))
                            dictin[res][atom]=-1.
        return dictin


class DSSR(PDB):
    
    def __init__(self,pdbfile):
        super(DSSR,self).__init__(pdbfile)
        
        def idfuc(x):
            x=list(x)
            if re.search('[A-Z]$',x[2]):
                x[2]=x[2][:-1]+'^'+x[2][-1]
            if re.search('\d$',x[0]):
                x[0]=x[0]+'/'
            out=x[1]+'.'+x[0]+x[2]
            return out
        
        self.id_res=dict(zip([idfuc(x) for x in self.xulie],[x[-1] for x in self.xulie]))
        
        pass
    
    def dssr(self):
        """
        perform x3dna-dssr program
        """
        
        pass
    
    def get_res_ss(self,dssrfile):
        """
        dssrfile => str (file)
        return: res_ss => list ['res1 str','res2 str',....]
        """
        with open(dssrfile,'r') as f:
            flist=f.readlines()
        flist=list(filter(lambda x:re.match('\s+\d+\s+\w\s+',x),flist))
        res_ss=list(map(lambda x:'\t'.join([self.id_res[x.split()[3]],x.split()[5]]),flist))
        return res_ss
    
    def get_ss_df(self,res_ss_file=None,dssrfile=None):
        """
        res_ss_file => str (file) or list
        dssrfile => str (file)
        eg. 
        row1: res1 stem
        row2: res2 haripin
        row3: res3 junction
        
        return => df (no.of res * 7)
        row => res col => ss_state
        """
        if dssrfile==None:
            with open(res_ss_file,'r') as f:
                flist=f.readlines()
        else:
            flist=self.get_res_ss(dssrfile)
        
        index=[x[-1] for x in self.xulie]
        columns=['bulge','ss-non-loop','stem','internal','hairpin','junction','pseudoknot']
        tmp=np.zeros((len(index),len(columns)),dtype=np.int)
        df=pd.DataFrame(tmp,index=index,columns=columns)
        
        for x in flist:
            for y in columns[:-1]:
                if re.search(y,x):
                    df.loc[x.split()[0],y]+=1
                    break
            if re.search(columns[-1],x):
                df.loc[x.split()[0],columns[-1]]+=1
        return df
    
    def get_ss_num(self,dssrfile):
        """
        dssrfile: str(file)
        return: df
        """
        def catnum(x,lines):
            num=0
            for l in lines:
                if not re.search('pseudoknot',x):
                    matobj=re.match(r'List of (\d+) '+x,l)
                else:
                    matobj=re.match(r'This structure contains (\d+)-order pseudoknot',l)
                if matobj:
                    num=int(matobj.group(1))
            return num
        
        with open(dssrfile,'r') as f:
            flist=f.readlines()
        
        index=['bulge','non-loop','stem','internal','hairpin','junction','pseudoknot']
        nums=[catnum(x,flist) for x in index]
        df=pd.Series(nums,index=index)
        return df


class DPX_CX(RSA):
    
    def __init__(self,pdbfile=None,asafile=None,**kwargs):
        self.pdbfile=pdbfile
        self.asafile=asafile
        super().__init__(asafile=asafile,pdbfile=pdbfile)
        
        self.res_atom_dpx=dict()
        self.res_atom_cx=dict()
        pass
    
    def dpx(self,mode='atom'):
        """
        mode => atom or residue
        return dict => {res1:{atom1:dpx},......}/{res1:dpx,......}
        missing atoms in asafile is assigned to value -1
        """
        super(DPX_CX,self).atom()
        self.res_atom_dpx=dict()
        
        asa_atoms=dict(map(lambda x:(x[0],list(filter(lambda y:float(x[1][y])>0,x[1]))),
                           self.res_atom_asa.items()))
        asa_atom_coords=list()
        for res in asa_atoms:
            asa_atom_coords.extend([self.coord[res][atom] for atom in asa_atoms[res]])
        
        for res in self.res_atom_asa:
            self.res_atom_dpx.setdefault(res,{})
            for atom in self.res_atom_asa[res]:
                if float(self.res_atom_asa[res][atom])>0:
                    self.res_atom_dpx[res][atom]=0.
                elif float(self.res_atom_asa[res][atom])<0:
                    self.res_atom_dpx[res][atom]=-1.
                else:
                    self.res_atom_dpx[res][atom]=min([self.cal_distance(self.coord[res][atom],x) for x in asa_atom_coords])
        if mode=='atom':
            return self.res_atom_dpx
        elif mode=='residue':
            return self._residue(self.res_atom_dpx)
        else:
            raise ValueError('not accepted mode attr')
        
        return
    
    def cx(self,mode='atom',R=10,atom_volume=20.1):
        """
        mode => atom or residue
        R => radius
        atom_volume => average volume of non-hydrogen atoms
        return dict => {res1:{atom1:cx,atom2:cx},.......}/{res1:cx}
        """
        df=super(DPX_CX,self)._atom_distance()
        vsphere=4/3*math.pi*pow(R,3)
        self.res_atom_cx=dict()
        for res in self.coord:
            self.res_atom_cx.setdefault(res,{})
            for atom in self.coord[res]:
                vint=len(np.where(df.loc[res+'_'+atom,:]<=R)[0])*atom_volume
                vext=vsphere-vint
                self.res_atom_cx[res][atom]=vext/vint
        if mode=='atom':
            return self.res_atom_cx
        elif mode=='residue':
            return self._residue(self.res_atom_cx)
        else:
            raise ValueError('not accepted mode attr')
        
        return
    
    def _residue(self,dictin):
        """
        to calculate residue level feature from atom level feature
        dictin: {res1:{atom1:value,value},......}
        """
        dictout=dict()
        for res in dictin:
            dictout[res]=np.mean(np.vstack(tuple(np.array(x[1]).reshape(1,-1) for x in dictin[res].items())),axis=0)
        return dictout
    
    def atom_array(self):
        """
        return: ndarray (number of atoms*2 (col.1 => dpx col.2 => cx))
        """
        if not self.res_atom_dpx:
            self.dpx(mode='atom')
        if not self.res_atom_cx:
            self.cx(mode='atom')
        return np.array(list(map(lambda x:[self.res_atom_dpx[x[0]][x[1]],self.res_atom_cx[x[0]][x[1]]],self.res_atom)))


class Laplacian(MPDB):
    
    def __init__(self,pdbfile=None):
        super(Laplacian,self).__init__(pdbfile)
        
    def _lap(self,distmap,coord):
        """
        distmap: ndarray
        coord: ndarray
        """
        distancelist=np.triu(distmap,k=1)
        distancelist=distancelist[np.nonzero(distancelist)].tolist()
        distancelist.sort()
        points1,points2=int(len(distancelist)/4),int(len(distancelist)/2)
        points3=points1+points2
        pointlist=[distancelist[i] for i in (0,points1,points2,points3,len(distancelist)-1)]
        
        omaplist=[]
        for point in pointlist:
            omap=np.exp(-np.square(distmap/point))
            omaplist.append(np.triu(omap,k=2)+np.tril(omap,k=-2))
        
        normlist=[]
        for omap in omaplist:
            olist=np.sum(omap,axis=1)
            res_normlist=list()
            for i in range(len(olist)):
                pi=coord[i,:]
                pj=np.sum(coord*omap[i,:].reshape(-1,1),axis=0)/olist[i]
                res_normlist.append(self.cal_distance(pi,pj))
            res_normlist=np.array(res_normlist)
            normlist.append(res_normlist)
        normlist=np.column_stack(tuple(normlist))
        return normlist
    
    def lap(self,mode='average'):
        """
        mode: average or atom name
        operators: 0,1/4,1/2,3/4,1
        return: {res1:[value1,value2,value3,value4,value5],........}
        """
        res_coord=self.get_res_coord_dict(mode=mode)
        coord=np.array(list(map(lambda x:res_coord[x[-1]],self.xulie)))
        distmap=self.distance_map(mode=mode)
        normlist=self._lap(distmap,coord)
        return dict(zip(np.array(self.xulie)[:,-1],normlist))
    
    def res_array(self,mode='average',normalize=True):
        """
        return: ndarray (number of res*5)
        """
        normdict=self.lap(mode)
        normlist=np.array(list(map(lambda x:normdict[x[-1]],self.xulie)))
        if normalize:
            normlist=(normlist-np.mean(normlist,axis=0))/np.std(normlist,axis=0)
        return normlist
    
    def atom_array(self,normalize=True):
        """
        return: ndarray (number of atoms*5)
        """
        coord=self.get_atom_coord_list()
        distmap=self._atom_distance().values
        normlist=self._lap(distmap,coord)
        if normalize:
            normlist=(normlist-np.mean(normlist,axis=0))/np.std(normlist,axis=0)
        return normlist


class Topo(MPDB):
    
    def __init__(self,pdbfile=None):
        super(Topo,self).__init__(pdbfile)
        self.contactlist=list()
        self.distancemap=None
        
    def contact(self,distcut,mode='min'):
        """
        mode: min,average,atom name
        self.contactlist: [(res1,res2),(res9,res2),......]
        """
        if not isinstance(self.distancemap,(np.ndarray,list)):
            contactmap=self.contact_map(distcut=distcut,mode=mode)
        else:
            contactmap=np.ones(self.distancemap.shape,dtype=np.int)
            contactmap[np.where(self.distancemap>distcut)]=0
            contactmap[np.diag_indices_from(contactmap)]=0
        ijindex=np.nonzero(contactmap)
        for i,j in zip(ijindex[0],ijindex[1]):
            self.contactlist.append([self.index_res[i],self.index_res[j]])
        return self.contactlist
            
    def topo(self,distcut=8,mode='min',array=False):
        """
        mode: min,average,atom name
        return: {res1:[de,clo,be,clu],........}
        """
        if not self.contactlist:
            self.contact(distcut=distcut,mode=mode)
        
        g=nx.Graph()
        g.add_edges_from(self.contactlist)
        degree=nx.degree(g)
        closeness=nx.closeness_centrality(g)
        betweenness=nx.betweenness_centrality(g)
        cluster=nx.clustering(g)
        
        outdict=dict()
        for key in self.xulie:
            res=key[-1]
            if res in [x[0] for x in degree]:
                outdict[res]=np.array([degree[res],closeness[res],betweenness[res],cluster[res]])
            else:
                outdict[res]=np.array([0.,0.,0.,0.])
        outlist=list()
        for key in self.xulie:
            outlist.append(outdict[key[-1]])
        if array:
            return np.array(outlist)
        return outdict
    
    def res_array(self,distcut,mode='min',normalize=True):
        array=self.topo(distcut=distcut,mode=mode,array=True)
        if normalize:
            array=(array-np.mean(array,axis=0))/np.std(array,axis=0)
        return array


class STRfeature(MPDB):
    
    def __init__(self,pdbfile,asafile=None,feature=['asa','topo','laps','dpx','cx'],
                 normlize=[False,True,True,False,False],
                 distcut=8,contactmode='min',lapsmode='average',prob=1.5):
        super().__init__(pdbfile)
        
        self.rsa=RSA(pdbfile=pdbfile,asafile=asafile,prob=prob)
        self.topo=Topo(pdbfile=pdbfile)
        self.laplacian=Laplacian(pdbfile=pdbfile)
        self.dpxcx=DPX_CX(pdbfile=pdbfile,asafile=asafile)
        
        self.featurelize=[
                      self.rsa.residue(relative=True),
                      self.topo.topo(distcut=distcut,mode=contactmode),
                      self.laplacian.lap(mode=lapsmode),
                      self.dpxcx.dpx(mode='residue'),
                      self.dpxcx.cx(mode='residue',R=10,atom_volume=20.1),
                          ]
        
        self.normlize=normlize
        
    def get_feature(self):
        """
        return: feature array (str) col.1=> res col.2,3,4....=>feature
        """
        reslist=np.array(self.xulie)[:,-1]
        featurelist=list(map(lambda x:self._sort_feature(x[0],x[1]),zip(self.featurelize,self.normlize)))
        return np.column_stack(tuple([reslist]+featurelist))
    
    def get_contact(self):
        """
        return: contact array (str) [(res1,res2),(res9,res2),......]
        """
        return np.array(self.topo.contactlist,dtype=str)
    
    def get_contact2(self):
        """
        return: contact array (int)
        """
        cont=self.get_contact()
        res_index=self.topo.res_index
        return np.array(list(map(lambda x:[res_index[x[0]],res_index[x[1]]],cont)),dtype=int)
        
    def _sort_feature(self,featuredict,normlize=False):
        """
        featuredict: {res:feature(list or float),......}
        return: array (float)
        """
        featurelist=list(map(lambda x:[x[0],x[1]],featuredict.items()))
        featurelist.sort(key=lambda x:np.array(self.xulie)[:,-1].tolist().index(x[0]))
        out=np.array(list(map(lambda x:x[1],featurelist)))
        if normlize:
            out=self.zscore(out)
        return out
    
    def zscore(self,data):
        data=np.array(data)
        mean=np.mean(data,axis=0)
        std=np.std(data,axis=0)
        return (data-mean)/std


class SEQfeature(PDB):
    
    def __init__(self,pdbfile):
        super().__init__(pdbfile)
        
        self.res_restype=dict(zip([x[-1] for x in self.xulie],[self.het_to_atom[x[0]] for x in self.xulie]))
    
    def get_trinucleotides_occurrence(self,windowsize):
        """
        windowsize: int
        return: dict {res:[occurrence,]*64,......}
        """
        tlist=['A','G','C','U']
        trip=list(map(lambda x:''.join(x),itertools.product(tlist,tlist,tlist)))
        return self.get_occurrence(mode='NA',windowsize=windowsize,target=trip)
    
    def get_occurrence(self,mode,windowsize,target):
        """
        mode: NA/protein
        windowsize: int
        target: str (eg. AC/IN/UU) / list
        return: dict {res:[occurrence,occurrence,...],...}
        """
        res_window=self.get_sequence_with_window(mode,windowsize)
        if isinstance(target,str):
            target=[target]
        
        for key in res_window:
            res_window[key]=[res_window[key].count(x) for x in target]
        return res_window
    
    def get_sequence_with_window(self,mode,windowsize):
        """
        mode: NA/protein
        windowsize: int
        return: dict {res:sequence,res:sequence,.......}
        """
        if mode not in ['NA','protein']:
            raise ValueError('not accepted mode')
        self.res_restype['NA']='-'
        
        res_window=self.get_res_seq_windows(windowsize=windowsize)
        for key in res_window:
            if mode=='NA':
                res_window[key]=''.join(list(map(lambda x:self.res_restype[x],res_window[key])))
            elif mode=='protein':
                res_window[key]=''.join(list(map(lambda x:self.three_to_one[self.res_restype[x]],res_window[key])))
        return res_window
    
    def pssm(self,pssmfile,windowsize=0,normlize=True):
        """
        pssmfile: str (file)
        windowsize: int
        return: dict {res:[fea,fea,fea,..],.......}
        """
        with open(pssmfile,'r') as f:
            flist=f.readlines()
        flist=list(filter(lambda x:re.match('\s+\d+',x),flist))
        fdict=dict(zip([x[-1] for x in self.xulie],[x.split()[2:22] for x in flist]))
        fdict['NA']=[0.]*20
        res_window=self.get_res_seq_windows(windowsize=windowsize)
        for key in res_window:
            res_window[key]=np.array(sum(list(map(lambda x:fdict[x],res_window[key])),[])).astype(np.float)
        if normlize:
            for key in res_window:
                res_window[key]=1/(1+np.exp(-res_window[key]))
        return res_window
    
    def hhm(self,hhmfile,windowsize=0,normlize=True):
        """
        hmmfile: str (file)
        windowsize: int
        NOTE: normlize is True in general
        return: dict {res:[fea,fea,fea,...],.......}
        """
        with open(hhmfile,'r') as f:
            flist=f.readlines()
        tag=list(filter(lambda x:re.match('#',x),flist))[0]
        index=flist.index(tag)+5
        hmmlist=[]
        while not re.match('//',flist[index]):
            hmmlist.append([-1000. if x=='*' else float(x) for x in flist[index].split()[2:-1]]\
                +[-1000. if x=='*' else float(x) for x in flist[index+1].split()])
            index+=3
        hmmdict=dict(zip([x[-1] for x in self.xulie],hmmlist))
        hmmdict['NA']=[-1000.]*30
        res_window=self.get_res_seq_windows(windowsize=windowsize)
        for key in res_window:
            res_window[key]=np.array(sum(list(map(lambda x:hmmdict[x],res_window[key])),[])).astype(np.float)
        if normlize:
            for key in res_window:
                tmp=2**(-0.001*res_window[key])
                tmp[np.where(tmp==2.)]=0.
                res_window[key]=tmp
        return res_window


class STAfeature(SEQfeature):
    
    def __init__(self,pdbfile):
        super().__init__(pdbfile)
        
        self.restypelist=[self.het_to_atom[x[0]] for x in self.xulie]
        
    def get_nucleotide_frequence(self):
        """
        return: dict {res:frequence,........}
        """
        nt_frequence=self.get_frequence(['A','G','C','U'])
        res_frequence=dict()
        for key in self.res_restype:
            res_frequence[key]=nt_frequence[self.res_restype[key]]
        return res_frequence
    
    def get_frequence(self,target):
        """
        target: str (eg. A/G/U) / list
        return: dict {target:frequence,......}
        """
        if isinstance(target,str):
            target=[target]
        
        return dict(zip(target,[self.restypelist.count(x)/len(self.restypelist) for x in target]))


if __name__=='__main__':
    # test=Laplacian(pdbfile=r'E:\deepbind\ligand\NABS\rna\1ddy_A.pdb')
    # feat=test.atom_array()
    pass