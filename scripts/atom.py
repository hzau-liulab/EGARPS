def ProA(r,a):
    if a=='CZ':
        if r=='ARG':
            return 0
        elif r in ['PHE','TYR']:
            return 4
        
    elif a=='CG':
        if r=='ASP':
            return 1
        elif r=='ASN':
            return 3
        elif r in ['HIS','PHE','TRP','TYR']:
            return 4
        elif r in ['ARG','GLN','GLU','LEU','LYS','PRO']:
            return 5
        elif r=='MET':
            return 7
        
    elif a=='CD':
        if r=='GLU':
            return 1
        elif r=='GLN':
            return 3
        elif r=='LYS':
            return 5
        elif r in ['ARG','PRO']:
            return 7
    
    elif a=='C':
        return 2
    
    elif a=='CD2':
        if r in ['HIS','PHE','TRP','TYR']:
            return 4
        elif r=='LEU':
            return 5
    
    elif a=='CE1':
        if r in ['HIS','PHE','TYR']:
            return 4
        
    elif a=='CD1':
        if r in ['PHE','TRP','TYR']:
            return 4
        elif r in ['ILE','LEU']:
            return 5
    
    elif a=='CE2':
        if r in ['PHE','TRP','TYR']:
            return 4
    
    elif a=='CE3':
        if r=='TRP':
            return 4
    
    elif a=='CH2':
        if r=='TRP':
            return 4
    
    elif a=='CZ2':
        if r=='TRP':
            return 4
    
    elif a=='CZ3':
        if r=='TRP':
            return 4
    
    elif a=='CB':
        if r in ['ALA','ARG','ASN','ASP','GLN','GLU','HIS','ILE','LEU','LYS','MET','PHE','PRO','TRP','TYR','VAL']:
            return 5
        elif r in ['CYS','SER','THR']:
            return 7
    
    elif a=='CG1':
        if r in ['ILE','VAL']:
            return 5
    
    elif a=='CG2':
        if r in ['ILE','THR','VAL']:
            return 5
    
    elif a=='CE':
        if r in ['LYS','MET']:
            return 7
    
    elif a=='N':
        if r in ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']:
            return 8
    
    elif a=='NH1':
        if r=='ARG':
            return 9
    
    elif a=='NH2':
        if r=='ARG':
            return 9
    
    elif a=='ND2':
        if r=='ASN':
            return 10
    
    elif a=='NE2':
        if r=='GLN':
            return 10
        elif r=='HIS':
            return 11
    
    elif a=='ND1':
        if r=='HIS':
            return 11
    
    elif a=='NE1':
        if r=='TRP':
            return 11
    
    elif a=='NE':
        if r=='ARG':
            return 12
    
    elif a=='NZ':
        if r=='LYS':
            return 13
    
    elif a=='O':
        return 14
    
    elif a=='OD1':
        if r=='ASN':
            return 15
        elif r=='ASP':
            return 17
    
    elif a=='OE1':
        if r=='GLN':
            return 15
        elif r=='GLU':
            return 17
    
    elif a=='OG':
        if r=='SER':
            return 16
    
    elif a=='OG1':
        if r=='THR':
            return 16
    
    elif a=='OH':
        if r=='TYR':
            return 16
    
    elif a=='OD2':
        if r=='ASP':
            return 17
    
    elif a=='OE2':
        if r=='GLU':
            return 17
    
    elif a=='SG':
        if r=='CYS':
            return 18
    
    elif a=='SD':
        if r=='MET':
            return 19
    
    else:
        return None

def RnaA(r,a):
    if a=='C2':
        if r in ['C','U']:
            return 0
        elif r in ['A','G']:
            return 1
    
    elif a=='C6':
        if r=='G':
            return 0
        elif r in ['A','C','U']:
            return 1
    
    elif a=='C4':
        if r=='U':
            return 0
        elif r in ['A','G','C']:
            return 1
    
    elif a=='C5':
        return 1
    
    elif a=='C8':
        if r in ['A','G']:
            return 1
    
    elif a in ["C1'","C2'","C3'","C4'","C5'"]:
        return 2
    
    elif a=='N1':
        if r in ['C','G','U']:
            return 3
        elif r=='A':
            return 5
    
    elif a=='N3':
        if r=='U':
            return 3
        elif r in ['A','G','C']:
            return 5
    
    elif a=='N6':
        if r=='A':
            return 4
    
    elif a=='N4':
        if r=='C':
            return 4
    
    elif a=='N2':
        if r=='G':
            return 4
    
    elif a=='N7':
        if r in ['A','G']:
            return 5
    
    elif a=='N9':
        if r in ['A','G']:
            return 6
    
    elif a=='O2':
        if r in ['C','U']:
            return 7
    
    elif a=='O6':
        if r=='G':
            return 7
    
    elif a=='O4':
        if r=='U':
            return 7
    
    elif a=="O2'":
        return 8
    
    elif a in ["O3'","O4'","O5'"]:
        return 9
    
    elif a in ['OP1','OP2']:
        return 10
    
    elif a=='P':
        return 11
    
    else:
        return None

def AtomPair(r1,a1,r2,a2):
    """
    interaction types according to ITScorePR/DITScorePR
    """
    return ProA(r1,a1), RnaA(r2,a2)

def Potential(data):
    with open(data, 'r') as f:
        fl = f.readlines()
    p = [float(l.split()[1]) for l in fl]
    p = [0.] + p
    return p


if __name__ == '__main__':
    import os,pickle
    
    pro_symbol=['C2+','C2-','C2M','C2S','Car','C3C','C3A','C3X','N2N','N2+',
                'N2X','Nar','N21','N3+','O2M','O2S','O3H','O2-','S31','S30']
    rna_symbol=['C2X','Car','C3X','N2N','N2X','Nar','N21','O2','O31','O32',
                'O2-','P']
    i=0
    potential=dict()
    for ps in pro_symbol:
        for rs in rna_symbol:
            pot_f='../DITScorePR/potentials/{}_{}.data'.format(ps,rs)
            if os.path.exists(pot_f):
                potential[i]=Potential(pot_f)
            else:
                print(ps,rs)
                potential[i]=[0.] * 46
            i+=1
    with open('../training_testing_set/residue_feature/potential.pkl','wb') as f:
        pickle.dump(potential, f)
    