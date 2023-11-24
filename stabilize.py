import sys
#sys.path.append('./RAFT/core')

import warnings
warnings.filterwarnings("ignore")
#from raft import RAFT

from pathlib import Path
from tqdm import tqdm

import gdown
import os

from glob import glob
import numpy as np
import torch

import cv2

from skimage import registration



from joblib import Parallel, delayed


#### DATA loading
def check_out_dic(path,out_path="out/"):
    out_path = out_path+path.split("/")[-2]+"/"
    Path(out_path).mkdir(exist_ok=True,parents=True)
    return out_path

def check_model(model_path):
    if os.path.isfile(model_path) == False:
        """The model referenced is a newer verion
           and somehow not compatible ... model --> ordered dict"""

        print("Downloading model")
        model_url = "https://drive.google.com/file/d/1fubTHIa_b2C8HqfbPtKXwoRd9QsYxRL6/view?usp=share_link"
        gdown.download(model_url, model_path, quiet=False,fuzzy=True)
    else:
        print("model found")

def get_source(path):
    if os.path.isdir(path):
        IMLIST = sorted(glob(path+"*.*"))
        return [IMLIST, False, len(IMLIST)]
    else:
        video = cv2.VideoCapture(path)
        return [video, True, video.get(cv2.CAP_PROP_FPS)]

def frame_preprocess(frame, device):

    if len(frame.shape) == 2:
        frame = frame/frame.max()*255
        frame = frame.astype("uint8")

        frame = np.dstack([frame]*3)

    frame = torch.from_numpy(frame.copy()).permute(2, 0, 1).type(torch.float32)
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame

def get_frame(IMOBJCT, IND):
    if IMOBJCT[1]==False:
        IMLIST = IMOBJCT[0]
        return cv2.imread(IMLIST[IND])[:,:,::-1]
    else:
        video = IMOBJCT[0]
        video.set(cv2.CAP_PROP_POS_FRAMES, IND)
        return video.read()[1][:,:,::-1]

#### estimate TRAFO
def estimate_trafo(IMOBJCT,IND,model,iters=11,device="cuda:0"):

    frame1 = get_frame(IMOBJCT,IND  )
    frame2 = get_frame(IMOBJCT,IND+1)

    frame1 = frame_preprocess(frame1, device=device)
    frame2 = frame_preprocess(frame2, device=device)

    flow_up = model(frame1, frame2,
                    iters=iters,
                    test_mode=True)[1].detach().cpu().numpy()[0]
    dx ,dy  = flow_up
    ddy,ddx = np.median(dy),np.median(dx)
    return [ddy, ddx]


def get_trafos_seriel(IMOBJCT,model,dev="cuda:0",iters=11):
    DYX = []
    for IND in tqdm(range(IMOBJCT[2]-1)):
        DY,DX = estimate_trafo(IMOBJCT,IND,model,iters=iters,device=dev)
        DYX.append([DY,DX])
    return np.stack(DYX)
#### apply Trafo



#### write data output

if __name__ == '__main__':

    dev        = "cpu" ## "cuda:0"
    iters      = 11
    path       = "samples/F9-1(MIT14v2-4ng)"
    model_path = "MODELS/raftsintel.pth"

    if path[-1] != "/":
        path += "/"

    out_path = check_out_dic(path)


    #check_model(model_path)

    print("Loading model:")
    model = torch.load(model_path,
                       map_location=torch.device(dev))

    IMOBJCT = get_source(path)

    print("Estimating displacement")


    DYX = get_trafos_seriel(IMOBJCT,model,dev=dev,iters=iters)
    #print(DYX)

    np.savetxt(out_path+"dyx.txt", DYX)




            #fps     = IMOBJCT.get(cv2.CAP_PROP_FPS)
