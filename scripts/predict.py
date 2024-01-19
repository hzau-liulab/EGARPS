import torch
import dgl
import pickle
import os
import numpy as np
from Models_res import Graph_Model
os.environ['DGLBACKEND'] = 'pytorch'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ins import get_input_data

model=Graph_Model()
model.load_state_dict(torch.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../weights/weights.pth')
    ))
model.to('cuda')
model.eval()

def predict_for_one(pdb):
    indata=get_input_data(pdb)
    with open(indata,'rb') as f:
        bgs,_,tags=pickle.load(f)
    assert len(bgs)==1

    with torch.no_grad():
        predictions=list()
        for bg in bgs:
            bg=[x.to('cuda') for x in bg]
            _,prediction=model(bg)
            prediction=torch.sigmoid(prediction).to('cpu').numpy()
            predictions.append(prediction)
        predictions=np.row_stack(predictions)
        
        with open(os.path.join(os.path.dirname(pdb),'result.txt'),'w') as f:
            np.savetxt(f,np.column_stack((tags[:1],predictions)),fmt='%s',delimiter='\t')

if __name__=='__main__':
    import sys
    
    print('Predicting for {}'.format(os.path.basename(sys.argv[1])))
    
    try:
        predict_for_one(sys.argv[1])
        print('{} successful predicted'.format(os.path.basename(sys.argv[1])))
    except:
        print('{} failure predicted'.format(os.path.basename(sys.argv[1])))