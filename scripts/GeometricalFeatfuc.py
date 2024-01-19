import os
import numpy as np


class MathMorphologyPocket(object):
    def __init__(self,pdbfile=None,ghecomresfile=None):
        self.pdb=pdbfile
        self.ghecomres=ghecomresfile
    
    def ghecom(self,pdbfile=None):
        """
        excute ghecom program
        """
        pdb=pdbfile if pdbfile is not None else self.pdb
        resout=''.join(pdb,'.txt')
        os.system('ghecom -M M -atmhet B -hetpep2atm F -ipdb '+pdb+' -ores '+resout)
        
    def descriptor(self,ghecomresfile=None):
        """
        return: strarray col.1=>res col.2=>shellAcc col.3=>Rinacc col.4=>pocketness
        """
        ghecomres=ghecomresfile if ghecomresfile is not None else self.ghecomres
        out=np.loadtxt(ghecomres,skiprows=43,usecols=(0,3,4,7),dtype=str)
        return out
    
    def res_array(self,ghecomresfile=None):
        return self.descriptor(ghecomresfile)
