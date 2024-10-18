import torch
import dgl
import pickle
import os
import numpy as np
from functools import partial
from Models_res import Graph_Model
os.environ['DGLBACKEND'] = 'pytorch'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ins import get_input_data


def load_model(version):
    model=Graph_Model(mid=version)
    model.load_state_dict(torch.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../weights/{}.pth'.format(version)),
        map_location='cpu'))
    model.to('cuda')
    model.eval()
    return model

def predict_for_one(pdb, academic, ws):
    indata=get_input_data(pdb, academic)
    with open(indata,'rb') as f:
        bgs,_,tags=pickle.load(f)
    assert len(bgs)==1

    predictions=list()
    for w in ws:
        model=load_model(w)
        with torch.no_grad():
            for bg in bgs:
                bg=[x.to('cuda') for x in bg]
                _,prediction=model(bg)
                predictions.append(prediction.to('cpu'))
    predictions=torch.cat(predictions, dim=1)
    
    predictions=torch.sigmoid(torch.mean(predictions)).numpy()
    with open(os.path.join(os.path.dirname(pdb),'result.txt'),'w') as f:
        np.savetxt(f,np.column_stack((tags[:1],predictions)),fmt='%s',delimiter='\t')

def parse_arg(*args):
    
    if '-non-academic' in args:
        if '-mini' in args:
            raise ValueError('Not accept mini togather with non-academic')
        else:
            predict=partial(predict_for_one, academic=False, ws=['a1w1','a1w2','a1w3'])
    else:
        if '-mini' in args:
            predict=partial(predict_for_one, academic=True, ws=['m1w1'])
        else:
            predict=partial(predict_for_one, academic=True, ws=['p1w1','p1w2','p1w3'])
    return predict

if __name__=='__main__':
    import sys
    
    print('Predicting for {}'.format(os.path.basename(sys.argv[-1])))
    
    predict = parse_arg(*sys.argv[:-1])
    
    try:
        predict(sys.argv[-1])
        print('{} successful predicted'.format(os.path.basename(sys.argv[-1])))   
    
    except:
        print('{} failure predicted'.format(os.path.basename(sys.argv[-1])))