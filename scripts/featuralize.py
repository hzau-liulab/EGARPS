from Bio.PDB.PDBParser import PDBParser
import numpy as np
from ModifyName import modify_residue_atom_name

""" atom name """
ATOMS_NAME = ['P', 'OP1', 'OP2', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'',
              'O4\'', 'C1\'', 'C2\'', 'O2\'', 'N9', 'C8', 'N7', 'C5', 'C6',
              'O6', 'N1', 'C2', 'N2', 'N3', 'C4', 'N6', 'O4', 'O2', 'N4',
              'N', 'CB', 'CG1', 'CG2', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'OD1', 'OD2', 'ND2',
              'NE', 'CZ', 'NH1', 'NH2', 'CD1', 'CD2', 'CE1', 'CE2', 'NE1', 'CZ2', 'CH2', 'CZ3', 'CE3',
              'OG', 'SG', 'SD', 'CE', 'OE1', 'NE2', 'OG1', 'NZ', 'ND1', 'C', 'O',
              'CA','OE2','OH']

""" atom mass """
MP = 1.1478706861464498#30.9738
MO = -0.5774529551825434#15.9994
MC = -1.0370238500543882#12.0107
MN = -0.8070482925634256#14.0067
MS = 1.2736544116539066#32.0655

""" atom charge """
QP = 1.1662
QOP1 = -0.776
QOP2 = -0.776
QO5S_1 = -0.4989
QO5S_2 = -0.6223
QC5S_A = 0.0558
QC5S_U = 0.0558
QC5S_G = 0.0558
QC5S_C = 0.0558
QC4S = 0.1065
QC3S = 0.2022
QO3S_1 = -0.5246
QO3S_2 = -0.6541
QO4S = -0.3548
QC1S_A = 0.0394
QC1S_G = 0.0191
QC1S_U = 0.0674
QC1S_C = 0.0066
QC2S = 0.067
QO2S = -0.6139
QN9_A = -0.0251
QN9_G = 0.0492
QC8_A = 0.2006
QC8_G = 0.1374
QN7_A = -0.6073
QN7_G = -0.5709
QC5_A = 0.0515
QC5_G = 0.1744
QC5_U = -0.3635
QC5_C = -0.5215
QC6_A = 0.7009
QC6_G = 0.477
QC6_U = -0.1126
QC6_C = 0.0053
QO6 = -0.5597
QN1_A = -0.7615
QN1_G = -0.4787
QN1_U = 0.0418
QN1_C = -0.0484
QC2_A = 0.5875
QC2_G = 0.7657
QC2_U = 0.4687
QC2_C = 0.7538
QN2 = -0.9672
QN3_A = -0.6997
QN3_G = -0.6323
QN3_U = -0.3549
QN3_C = -0.7584
QC4_A = 0.3053
QC4_G = 0.1222
QC4_U = 0.5952
QC4_C = 0.8185
QN6 = -0.9019
QO4 = -0.5761
QO2_U = -0.5477
QO2_C = -0.6252
QN4 = -0.953

#protein
QN_VAL = -0.4157
QC1S_VAL = -0.0875
QCB_VAL = 0.2985
QCG1_VAL = -0.3192
QCG2_VAL = -0.3192
QC_VAL = 0.5973
QO_VAL = -0.5679
QN_ALA = -0.4157
QC1S_ALA = 0.0337
QCB_ALA = -0.1825
QC_ALA = 0.5973
QO_ALA = -0.5679
QN_ARG = -0.3479
QC1S_ARG = -0.2637
QCB_ARG = -0.0007
QCG_ARG = 0.0390
QCD_ARG = 0.0486
QNE_ARG = -0.5295
QCZ_ARG = 0.8076
QNH1_ARG = -0.8627
QNH2_ARG = -0.8627
QC_ARG = 0.7341
QO_ARG = -0.5894
QN_ASP = -0.5163
QC1S_ASP = 0.0381
QCB_ASP = -0.0303
QCG_ASP = 0.7994
QOD1_ASP = -0.8014
QOD2_ASP = -0.8014
QC_ASP = 0.5366
QO_ASP = -0.5819
QN_ASN = -0.4157
QC1S_ASN = 0.0143
QCB_ASN = -0.2041
QCG_ASN = 0.7130
QOD1_ASN = -0.5931
QND2_ASN = -0.9191
QC_ASN = 0.5973
QO_ASN = -0.5679
QN_LEU = -0.4157
QC1S_LEU = -0.0518
QCB_LEU = -0.1102
QCG_LEU = 0.3531
QCD1_LEU = -0.4121
QCD2_LEU = -0.4121
QC_LEU = 0.5973
QO_LEU = -0.5679
QN_ILE = -0.4157
QC1S_ILE = -0.0597
QCB_ILE = 0.1303
QCG2_ILE = -0.3204
QCG1_ILE = -0.0430
QCD1_ILE = -0.0660
QC_ILE = 0.5973
QO_ILE = -0.5679
QN_PHE = -0.4157
QC1S_PHE = -0.0024
QCB_PHE = -0.0343
QCG_PHE = 0.0118
QCD1_PHE = -0.1256
QCE1_PHE = -0.1704
QCZ_PHE = -0.1072
QCE2_PHE = -0.1704
QCD2_PHE = -0.1256
QC_PHE = 0.5973
QO_PHE = -0.5679
QN_PRO = -0.2548
QCD_PRO =0.0192
QCG_PRO = -0.0189
QCB_PRO = -0.0070
QC1S_PRO = -0.0266
QC_PRO = 0.05896
QO_PRO = -0.5748
QN_TRP = -0.4157
QC1S_TRP = -0.0275
QCB_TRP = -0.0050
QCG_TRP = -0.1415
QCD1_TRP = -0.1638
QNE1_TRP = -0.3418
QCE2_TRP = 0.1380
QCZ2_TRP = -0.2601
QCH2_TRP = -0.1134
QCZ3_TRP = -0.1972
QCE3_TRP = -0.2387
QCD2_TRP = 0.1243
QC_TRP = 0.5973
QO_TRP = -0.5679
QN_SER = -0.4157
QC1S_SER = -0.0249
QCB_SER = 0.2117
QOG_SER = -0.6546
QC_SER = 0.5973
QO_SER = -0.5679
QN_TYR = -0.4157
QC1S_TYR = -0.0014
QCB_TYR = -0.0152
QCG_TYR = -0.0011
QCD1_TYR = -0.1906
QCE1_TYR = -0.2341
QCZ_TYR = 0.3226
QCE2_TYR = -0.2341
QCD2_TYR = -0.1906
QC_TYR = 0.5973
QO_TYR = -0.5679
QN_CYS = -0.4157
QC1S_CYS = 0.0213
QCB_CYS = -0.1231
QSG_CYS = -0.3119
QC_CYS = 0.5973
QO_CYS = -0.5679
QN_MET = -0.4157
QC1S_MET = -0.0237
QCB_MET = 0.0342
QCG_MET = 0.0018
QSD_MET = -0.2737
QCE_MET = -0.0536
QC_MET = 0.5973
QO_MET = -0.5679
QN_GLN = -0.4157
QC1S_GLN = -0.0031
QCB_GLN = -0.0036
QCG_GLN = -0.0645
QCD_GLN = 0.6951
QOE1_GLN = -0.6086
QNE2_GLN = -0.9407
QC_GLN = 0.5973
QO_GLN = -0.5679
QN_THR = -0.4157
QC1S_THR = -0.0389
QCB_THR = 0.3654
QCG2_THR = -0.2438
QOG1_THR = -0.6761
QC_THR = 0.5973
QO_THR = -0.5679
QN_GLU = -0.5163
QC1S_GLU = 0.0397
QCB_GLU = 0.0560
QCG_GLU = 0.0136
QCD_GLU = 0.8054
QOE1_GLU = -0.8188
QOE2_GLU = -0.8188
QC_GLU = 0.5366
QO_GLU = -0.5819
QN_LYS = -0.3479
QC1S_LYS = -0.2400
QCB_LYS = -0.0094
QCG_LYS = -0.187
QCD_LYS = -0.0479
QCE_LYS = -0.0143
QNZ_LYS = -0.3854
QC_LYS = 0.7341
QO_LYS = -0.5894
QN_HIS = -0.4157
QC1S_HIS = -0.0581
QCB_HIS = -0.0074
QCG_HIS = 0.1868
QND1_HIS = -0.5432
QCE1_HIS = 0.1635
QNE2_HIS = -0.2795
QCD2_HIS = -0.2207
QC_HIS = 0.5973
QO_HIS = -0.5679
QN_GLY = -0.4157
QC1S_GLY = -0.0252
QC_GLY = 0.5973
QO_GLY = -0.5679
QN_ASH = -0.4157
QC1S_ASH = 0.0341
QCB_ASH = -0.0316
QCG_ASH = 0.6462
QOD1_ASH = -0.5554
QOD2_ASH = -0.6376
QC_ASH = 0.5973
QO_ASH = -0.5679
QN_CYM = -0.4157
QC1S_CYM = -0.0351
QCB_CYM = -0.2413
QSG_CYM = -0.8844
QC_CYM = 0.5973
QO_CYM = -0.5679
QN_CYX = -0.4157
QC1S_CYX = 0.0429
QCB_CYX = -0.0790
QSG_CYX = -0.1081
QC_CYX = 0.5973
QO_CYX = -0.5679
QN_GLH = -0.4157
QC1S_GLH = 0.0145
QCB_GLH = -0.0071
QCG_GLH = -0.0174
QCD_GLH = 0.6801
QOE1_GLH = -0.5838
QOE2_GLH = -0.6511
QC_GLH = 0.5973
QO_GLH = -0.5679
QN_LYN = -0.4157
QC1S_LYN = -0.0721
QCB_LYN = -0.0484
QCG_LYN = 0.0661
QCD_LYN = -0.0377
QCE_LYN = 0.3260
QNZ_LYN = -1.0358
QC_LYN = 0.5973
QO_LYN = -0.5679
QN_HID = -0.4157
QC1S_HID = 0.0188
QCB_HID = -0.0462
QCG_HID = -0.0266
QND1_HID = -0.3811
QCE1_HID = 0.2057
QNE2_HID = -0.5727
QCD2_HID = 0.1292
QC_HID = 0.5973
QO_HID = -0.5679
QN_HIE = -0.4157
QC1S_HIE = -0.0581
QCB_HIE = -0.0074
QCG_HIE = 0.1868
QND1_HIE = -0.5432
QCE1_HIE = 0.1635
QNE2_HIE = -0.2795
QCD2_HIE = -0.2207
QC_HIE = 0.5973
QO_HIE = -0.5679
QN_HIP = -0.3479
QC1S_HIP = -0.1354
QCB_HIP = -0.4140
QCG_HIP = -0.0012
QND1_HIP = -0.1513
QCE1_HIP = -0.0170
QNE2_HIP = -0.1718
QCD2_HIP = -0.1141
QC_HIP = 0.7341
QO_HIP = -0.5894

#added
QCA_ALA = 0.0337
QCA_ARG = -0.2637
QCA_ASH = 0.0341
QCA_ASN = 0.0143
QCA_ASP = 0.0381
QCA_CYM = -0.0351
QCA_CYS = 0.0213
QCA_CYX = 0.0429
QCA_GLH = 0.0145
QCA_GLN = -0.0031
QCA_GLU = 0.0397
QCA_GLY = -0.0252
QCA_HID = 0.0188
QCA_HIE = -0.0581
QCA_HIP = -0.1354
QCA_ILE = -0.0597
QCA_LEU = -0.0518
QCA_LYN = -0.07206
QCA_LYS = -0.24
QCA_MET = -0.0237
QCA_PHE = -0.0024
QCA_PRO = -0.0266
QCA_SER = -0.0249
QCA_THR = -0.0389
QCA_TRP = -0.0275
QCA_TYR = -0.0014
QCA_VAL = -0.0875
QCA_HIS = 0.00

QOH_TYR = -0.5579


DICT_MASS = {'P': MP, 'OP1': MO, 'OP2': MO, 'O5\'': MO, 'C5\'': MC, 'C4\'': MC,
             'C3\'': MC, 'O3\'': MO, 'O4\'': MO, 'C1\'': MC, 'C2\'': MC,
             'O2\'': MO, 'N9': MN, 'C8': MC, 'N7': MN, 'C5': MC, 'C6': MC,
             'O6': MO, 'N1': MN, 'C2': MC, 'N2': MN, 'N3': MN, 'C4': MC,
             'N6': MN, 'O4': MO, 'O2': MO, 'N4': MN,
             'N': MN, 'CB': MC, 'CG1': MC, 'CG2': MC, 'CG': MC, 'CD': MC,
             'NE': MN, 'CZ': MC, 'NH1': MN, 'NH2': MN, 'OD1': MO, 'OD2': MO, 'ND2': MN,
             'CD1': MC, 'CD2': MC, 'CE1': MC, 'CE2': MC, 'NE1': MN, 'CZ2': MC, 'CH2': MC,
             'CZ3': MC, 'CE3': MC, 'OG': MO, 'SG': MS, 'SD': MS, 'CE': MC, 'OE1': MO,
             'NE2': MN, 'OG1': MO, 'OE2': MO, 'NZ': MN, 'ND1': MN, 'C': MC, 'O': MO,
             'CA': MC, 'OH': MO}

DICT_CHARGE = {'P': QP, 'OP1': QOP1, 'OP2': QOP2, 'C4\'': QC4S,
               'C3\'': QC3S, 'O4\'': QO4S, 'C2\'': QC2S, 'O2\'': QO2S,
               'O6': QO6, 'N2': QN2, 'N6': QN6, 'O4': QO4, 'N4': QN4}

DICT_CHARGE_O5S = {'nonhead': QO5S_1, 'head': QO5S_2}

DICT_CHARGE_O = {'VAL': QO_VAL, 'ALA': QO_ALA, 'ARG': QO_ARG, 'ASP': QO_ASP, 'ASN': QO_ASN, 'LEU': QO_LEU,
                   'ILE': QO_ILE, 'PHE': QO_PHE, 'PRO': QO_PRO, 'TRP': QO_TRP, 'SER': QO_SER, 'TYR': QO_TYR,
                   'CYS': QO_CYS, 'MET': QO_MET, 'GLN': QO_GLN, 'THR': QO_THR, 'GLU': QO_GLU, 'LYS': QO_LYS,
                   'HIS': QO_HIS, 'GLY': QO_GLY, 'ASH': QO_ASH, 'CYM': QO_CYM, 'CYX': QO_CYX, 'GLH': QO_GLH,
                   'LYN': QO_LYN, 'HID': QO_HID, 'HIE': QO_HIE, 'HIP': QO_HIP}

DICT_CHARGE_C5S = {'  A': QC5S_A, '  U': QC5S_U, '  C': QC5S_C, '  G': QC5S_G, 'A': QC5S_A, 'U': QC5S_U, 'C': QC5S_C, 'G': QC5S_G,}

DICT_CHARGE_C = {'VAL': QC_VAL, 'ALA': QC_ALA, 'ARG': QC_ARG, 'ASP': QC_ASP, 'ASN': QC_ASN, 'LEU': QC_LEU,
                   'ILE': QC_ILE, 'PHE': QC_PHE, 'PRO': QC_PRO, 'TRP': QC_TRP, 'SER': QC_SER, 'TYR': QC_TYR,
                   'CYS': QC_CYS, 'MET': QC_MET, 'GLN': QC_GLN, 'THR': QC_THR, 'GLU': QC_GLU, 'LYS': QC_LYS,
                   'HIS': QC_HIS, 'GLY': QC_GLY, 'ASH': QC_ASH, 'CYM': QC_CYM, 'CYX': QC_CYX, 'GLH': QC_GLH,
                   'LYN': QC_LYN, 'HID': QC_HID, 'HIE': QC_HIE, 'HIP': QC_HIP}

DICT_CHARGE_CA = {'VAL': QCA_VAL, 'ALA': QCA_ALA, 'ARG': QCA_ARG, 'ASP': QCA_ASP, 'ASN': QCA_ASN, 'LEU': QCA_LEU,
                   'ILE': QCA_ILE, 'PHE': QCA_PHE, 'PRO': QCA_PRO, 'TRP': QCA_TRP, 'SER': QCA_SER, 'TYR': QCA_TYR,
                   'CYS': QCA_CYS, 'MET': QCA_MET, 'GLN': QCA_GLN, 'THR': QCA_THR, 'GLU': QCA_GLU, 'LYS': QCA_LYS,
                   'HIS': QCA_HIS, 'GLY': QCA_GLY, 'ASH': QCA_ASH, 'CYM': QCA_CYM, 'CYX': QCA_CYX, 'GLH': QCA_GLH,
                   'LYN': QCA_LYN, 'HID': QCA_HID, 'HIE': QCA_HIE, 'HIP': QCA_HIP}

DICT_CHARGE_O3S = {'nontail': QO3S_1, 'tail': QO3S_2}

DICT_CHARGE_C1S = {'  A': QC1S_A, '  G': QC1S_G, '  U': QC1S_U, '  C': QC1S_C, 'A': QC1S_A, 'G': QC1S_G, 'U': QC1S_U, 'C': QC1S_C,
                   'VAL': QC1S_VAL, 'ALA': QC1S_ALA, 'ARG': QC1S_ARG, 'ASP': QC1S_ASP, 'ASN': QC1S_ASN, 'LEU': QC1S_LEU,
                   'ILE': QC1S_ILE, 'PHE': QC1S_PHE, 'PRO': QC1S_PRO, 'TRP': QC1S_TRP, 'SER': QC1S_SER, 'TYR': QC1S_TYR,
                   'CYS': QC1S_CYS, 'MET': QC1S_MET, 'GLN': QC1S_GLN, 'THR': QC1S_THR, 'GLU': QC1S_GLU, 'LYS': QC1S_LYS,
                   'HIS': QC1S_HIS, 'GLY': QC1S_GLY, 'ASH': QC1S_ASH, 'CYM': QC1S_CYM, 'CYX': QC1S_CYX, 'GLH': QC1S_GLH,
                   'LYN': QC1S_LYN, 'HID': QC1S_HID, 'HIE': QC1S_HIE, 'HIP': QC1S_HIP}

DICT_CHARGE_N9 = {'  A': QN9_A, '  G': QN9_G, 'A': QN9_A, 'G': QN9_G}

DICT_CHARGE_C8 = {'  A': QC8_A, '  G': QC8_G, 'A': QC8_A, 'G': QC8_G}

DICT_CHARGE_N7 = {'  A': QN7_A, '  G': QN7_G, 'A': QN7_A, 'G': QN7_G}

DICT_CHARGE_C5 = {'  A': QC5_A, '  G': QC5_G, '  U': QC5_U, '  C': QC5_C, 'A': QC5_A, 'G': QC5_G, 'U': QC5_U, 'C': QC5_C}

DICT_CHARGE_C6 = {'  A': QC6_A, '  G': QC6_G, '  U': QC6_U, '  C': QC6_C, 'A': QC6_A, 'G': QC6_G, 'U': QC6_U, 'C': QC6_C}

DICT_CHARGE_N1 = {'  A': QN1_A, '  G': QN1_G, '  U': QN1_U, '  C': QN1_C, 'A': QN1_A, 'G': QN1_G, 'U': QN1_U, 'C': QN1_C}

DICT_CHARGE_C2 = {'  A': QC2_A, '  G': QC2_G, '  U': QC2_U, '  C': QC2_C, 'A': QC2_A, 'G': QC2_G, 'U': QC2_U, 'C': QC2_C}

DICT_CHARGE_N3 = {'  A': QN3_A, '  G': QN3_G, '  U': QN3_U, '  C': QN3_C, 'A': QN3_A, 'G': QN3_G, 'U': QN3_U, 'C': QN3_C}

DICT_CHARGE_C4 = {'  A': QC4_A, '  G': QC4_G, '  U': QC4_U, '  C': QC4_C, 'A': QC4_A, 'G': QC4_G, 'U': QC4_U, 'C': QC4_C}

DICT_CHARGE_O2 = {'  U': QO2_U, '  C': QO2_C, 'U': QO2_U, 'C': QO2_C}

DICT_CHARGE_N = {'VAL': QN_VAL, 'ALA': QN_ALA, 'ARG': QN_ARG, 'ASP': QN_ASP, 'ASN': QN_ASN, 'LEU': QN_LEU,
                  'ILE': QN_ILE, 'PHE': QN_PHE, 'PRO': QN_PRO, 'TRP': QN_TRP, 'SER': QN_SER, 'TYR': QN_TYR,
                  'CYS': QN_CYS, 'MET': QN_MET, 'GLN': QN_GLN, 'THR': QN_THR, 'GLU': QN_GLU, 'LYS': QN_LYS,
                  'HIS': QN_HIS, 'GLY': QN_GLY, 'ASH': QN_ASH, 'CYM': QN_CYM, 'CYX': QN_CYX, 'GLH': QN_GLH,
                  'LYN': QN_LYN, 'HID': QN_HID, 'HIE': QN_HIE, 'HIP': QN_HIP}

DICT_CHARGE_CB = {'VAL': QCB_VAL, 'ALA': QCB_ALA, 'ARG': QCB_ARG, 'ASP': QCB_ASP, 'ASN': QCB_ASN, 'LEU': QCB_LEU,
                  'ILE': QCB_ILE, 'PHE': QCB_PHE, 'PRO': QCB_PRO, 'TRP': QCB_TRP, 'SER': QCB_SER, 'TYR': QCB_TYR,
                  'CYS': QCB_CYS, 'MET': QCB_MET, 'GLN': QCB_GLN, 'THR': QCB_THR, 'GLU': QCB_GLU, 'LYS': QCB_LYS,
                  'HIS': QCB_HIS, 'ASH': QCB_ASH, 'CYM': QCB_CYM, 'CYX': QCB_CYX, 'GLH': QCB_GLH, 'LYN': QCB_LYN,
                  'HID': QCB_HID, 'HIE': QCB_HIE, 'HIP': QCB_HIP}

DICT_CHARGE_CG1 = {'VAL': QCG1_VAL, 'ILE': QCG1_ILE}

DICT_CHARGE_CG2 = {'VAL': QCG2_VAL, 'ILE': QCG2_ILE, 'THR': QCG2_THR}

DICT_CHARGE_CG = {'ARG': QCG_ARG, 'ASP': QCG_ASP, 'ASN': QCG_ASN, 'LEU': QCG_LEU, 'PHE': QCG_PHE, 'PRO': QCG_PRO,
                  'TRP': QCG_TRP, 'TYR': QCG_TYR, 'MET': QCG_MET, 'GLN': QCG_GLN, 'GLU': QCG_GLU, 'LYS': QCG_LYS,
                  'HIS': QCG_HIS, 'ASH': QCG_ASH, 'GLH': QCG_GLH, 'LYN': QCG_LYN, 'HID': QCG_HID, 'HIE': QCG_HIE,
                  'HIP': QCG_HIP}

DICT_CHARGE_CD = {'ARG': QCD_ARG, 'PRO': QCD_PRO, 'GLN': QCD_GLN, 'GLU': QCD_GLU, 'LYS': QCD_LYS, 'GLH': QCD_GLH,
                  'LYN': QCD_LYN}

DICT_CHARGE_NE = {'ARG': QNE_ARG}

DICT_CHARGE_CZ = {'ARG': QCZ_ARG, 'PHE': QCZ_PHE, 'TYR': QCZ_TYR}

DICT_CHARGE_NH1 = {'ARG': QNH1_ARG}

DICT_CHARGE_NH2 = {'ARG': QNH2_ARG}

DICT_CHARGE_OD1 = {'ASP': QOD1_ASP, 'ASN': QOD1_ASN, 'ASH': QOD1_ASH}

DICT_CHARGE_OD2 = {'ASP': QOD2_ASP, 'ASH': QOD2_ASH}

DICT_CHARGE_ND2 = {'ASN': QND2_ASN}

DICT_CHARGE_CD1 = {'LEU': QCD1_LEU, 'ILE': QCD1_ILE, 'PHE': QCD1_PHE, 'TRP': QCD1_TRP, 'TYR': QCD1_TYR}

DICT_CHARGE_CD2 = {'LEU': QCD2_LEU, 'PHE': QCD2_PHE, 'TRP': QCD2_TRP, 'TYR': QCD2_TYR, 'HIS': QCD2_HIS, 'HID': QCD2_HID,
                   'HIE': QCD2_HIE, 'HIP': QCD2_HIP}

DICT_CHARGE_CE1 = {'PHE': QCE1_PHE, 'TYR': QCE1_TYR, 'HIS': QCE1_HIS, 'HID': QCE1_HID, 'HIE': QCE1_HIE, 'HIP': QCE1_HIP}

DICT_CHARGE_CE2 = {'PHE': QCE2_PHE, 'TRP': QCE2_TRP, 'TYR': QCE2_TYR}

DICT_CHARGE_NE1 = {'TRP': QNE1_TRP}

DICT_CHARGE_CZ2 = {'TRP': QCZ2_TRP}

DICT_CHARGE_CH2 = {'TRP': QCH2_TRP}

DICT_CHARGE_CZ3 = {'TRP': QCZ3_TRP}

DICT_CHARGE_CE3 = {'TRP': QCE3_TRP}

DICT_CHARGE_OG = {'SER': QOG_SER}

DICT_CHARGE_SG = {'CYS': QSG_CYS, 'CYM': QSG_CYM, 'CYX': QSG_CYX}

DICT_CHARGE_SD = {'MET': QSD_MET}

DICT_CHARGE_CE = {'MET': QCE_MET, 'LYS': QCE_LYS, 'LYN': QCE_LYN}

DICT_CHARGE_OE1 = {'GLN': QOE1_GLN, 'GLU': QOE1_GLU, 'GLH': QOE1_GLH}

DICT_CHARGE_NE2 = {'GLN': QNE2_GLN, 'HIS': QNE2_HIS, 'HID': QNE2_HID, 'HIE': QNE2_HIE, 'HIP': QNE2_HIP}

DICT_CHARGE_OG1 = {'THR': QOG1_THR}

DICT_CHARGE_OE2 = {'GLU': QOE2_GLU, 'GLH': QOE2_GLH}

DICT_CHARGE_NZ = {'LYS': QNZ_LYS, 'LYN': QNZ_LYN}

DICT_CHARGE_ND1 = {'HIS': QND1_HIS, 'HID': QND1_HID, 'HIE': QND1_HIE, 'HIP': QND1_HIP}

DICT_CHARGE_OH = {'TYR': QOH_TYR}

def is_head_residue(residue_list, res_index, residue):
    """ if residue is the head residue:
            return True
        else:
            return False
    """
    if res_index == 0:
        return True
    elif residue.get_full_id()[2] !=\
            residue_list[res_index - 1].get_full_id()[2]:
        return True
    else:
        return False


def is_tail_residue(residue_list, res_index, residue):
    """ if residue is the tail residue:
            return True
        else:
            return False
    """
    if res_index == len(residue_list) - 1:
        return True
    elif residue.get_full_id()[2] !=\
            residue_list[res_index + 1].get_full_id()[2]:
        return True
    else:
        return False

def get_atom_mass_charge(model):
    residues = list(model.get_residues())
    modify_residue_atom_name(residues)
    
    outs=list()
    res_index2 = 0
    for residue2 in residues:
        res_name = residue2.get_resname()

        for atom in residue2:
            atom_name = atom.get_name()

            if atom_name not in ATOMS_NAME:
                print(atom_name)
                outs.append([0.,0.])
                continue

            atom_mass = DICT_MASS[atom_name]

            if atom_name in DICT_CHARGE:
                atom_charge = DICT_CHARGE[atom_name]
            elif atom_name == 'O5\'':
                if is_head_residue(residues, res_index2, residue2):
                    atom_charge = DICT_CHARGE_O5S['head']
                else:
                    atom_charge = DICT_CHARGE_O5S['nonhead']
            elif atom_name == 'O3\'':
                if is_tail_residue(residues, res_index2, residue2):
                    atom_charge = DICT_CHARGE_O3S['tail']
                else:
                    atom_charge = DICT_CHARGE_O3S['nontail']
            elif atom_name == 'C1\'':
                atom_charge = DICT_CHARGE_C1S[res_name]
            elif atom_name == 'N9':
                atom_charge = DICT_CHARGE_N9[res_name]
            elif atom_name == 'C8':
                atom_charge = DICT_CHARGE_C8[res_name]
            elif atom_name == 'N7':
                atom_charge = DICT_CHARGE_N7[res_name]
            elif atom_name == 'C5':
                atom_charge = DICT_CHARGE_C5[res_name]
            elif atom_name == 'C5\'':
                atom_charge = DICT_CHARGE_C5S[res_name]
            elif atom_name == 'C6':
                atom_charge = DICT_CHARGE_C6[res_name]
            elif atom_name == 'N1':
                atom_charge = DICT_CHARGE_N1[res_name]
            elif atom_name == 'C2':
                atom_charge = DICT_CHARGE_C2[res_name]
            elif atom_name == 'N3':
                atom_charge = DICT_CHARGE_N3[res_name]
            elif atom_name == 'C4':
                atom_charge = DICT_CHARGE_C4[res_name]
            elif atom_name == 'O2':
                atom_charge = DICT_CHARGE_O2[res_name]
            elif atom_name == 'N':
                atom_charge = DICT_CHARGE_N[res_name]
            elif atom_name == 'CB':
                atom_charge = DICT_CHARGE_CB[res_name]
            elif atom_name == 'CG1':
                atom_charge = DICT_CHARGE_CG1[res_name]
            elif atom_name == 'CG2':
                atom_charge = DICT_CHARGE_CG2[res_name]
            elif atom_name == 'CG':
                atom_charge = DICT_CHARGE_CG[res_name]
            elif atom_name == 'CD':
                atom_charge = DICT_CHARGE_CD[res_name]
            elif atom_name == 'NE':
                atom_charge = DICT_CHARGE_NE[res_name]
            elif atom_name == 'CZ':
                atom_charge = DICT_CHARGE_CZ[res_name]
            elif atom_name == 'NH1':
                atom_charge = DICT_CHARGE_NH1[res_name]
            elif atom_name == 'NH2':
                atom_charge = DICT_CHARGE_NH2[res_name]
            elif atom_name == 'OD1':
                atom_charge = DICT_CHARGE_OD1[res_name]
            elif atom_name == 'OD2':
                atom_charge = DICT_CHARGE_OD2[res_name]
            elif atom_name == 'ND2':
                atom_charge = DICT_CHARGE_ND2[res_name]
            elif atom_name == 'CD1':
                atom_charge = DICT_CHARGE_CD1[res_name]
            elif atom_name == 'CD2':
                atom_charge = DICT_CHARGE_CD2[res_name]
            elif atom_name == 'CE1':
                atom_charge = DICT_CHARGE_CE1[res_name]
            elif atom_name == 'CE2':
                atom_charge = DICT_CHARGE_CE2[res_name]
            elif atom_name == 'NE1':
                atom_charge = DICT_CHARGE_NE1[res_name]
            elif atom_name == 'CZ2':
                atom_charge = DICT_CHARGE_CZ2[res_name]
            elif atom_name == 'CH2':
                atom_charge = DICT_CHARGE_CH2[res_name]
            elif atom_name == 'CZ3':
                atom_charge = DICT_CHARGE_CZ3[res_name]
            elif atom_name == 'CE3':
                atom_charge = DICT_CHARGE_CE3[res_name]
            elif atom_name == 'OG':
                atom_charge = DICT_CHARGE_OG[res_name]
            elif atom_name == 'SG':
                atom_charge = DICT_CHARGE_SG[res_name]
            elif atom_name == 'SD':
                atom_charge = DICT_CHARGE_SD[res_name]
            elif atom_name == 'CE':
                atom_charge = DICT_CHARGE_CE[res_name]
            elif atom_name == 'OE1':
                atom_charge = DICT_CHARGE_OE1[res_name]
            elif atom_name == 'NE2':
                atom_charge = DICT_CHARGE_NE2[res_name]
            elif atom_name == 'OG1':
                atom_charge = DICT_CHARGE_OG1[res_name]
            elif atom_name == 'OE2':
                atom_charge = DICT_CHARGE_OE2[res_name]
            elif atom_name == 'NZ':
                atom_charge = DICT_CHARGE_NZ[res_name]
            elif atom_name == 'ND1':
                atom_charge = DICT_CHARGE_ND1[res_name]
            elif atom_name == 'O':
                atom_charge = DICT_CHARGE_O[res_name]
            elif atom_name == 'C':
                atom_charge = DICT_CHARGE_C[res_name]
            elif atom_name == 'CA':
                atom_charge = DICT_CHARGE_CA[res_name]
            elif atom_name == 'OH':
                atom_charge == DICT_CHARGE_OH[res_name]
            else:
                print(res_name,atom_name)
                atom_charge = 0.
            outs.append([atom_mass,atom_charge])
        res_index2 += 1
    return np.array(outs,dtype=float)

def featuralize(pdb):
    p=PDBParser(QUIET=True)
    s=p.get_structure('pdb',pdb)
    model=s[0]
    feats=get_atom_mass_charge(model)
    return feats
