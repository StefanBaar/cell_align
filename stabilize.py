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

from skimage import registration, transform

from joblib import Parallel, delayed


#### DATA loading
def check_out_dic(path,out_path="out/"):
    out_path = out_path+path.split("/")[-2]+"/"
    Path(out_path+"images/").mkdir(exist_ok=True,parents=True)
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
    DYX = [[0,0]]
    for IND in tqdm(range(IMOBJCT[2]-1)):
        DY,DX = estimate_trafo(IMOBJCT,IND,model,iters=iters,device=dev)
        DYX.append([DY,DX])
    return np.stack(DYX)
#### apply Trafo

def apply_trafo(image, dyx, order=5):
    move_tf  = transform.AffineTransform(translation=(dyx[1],dyx[0]))
    image_tf = transform.warp(image,
                              move_tf.inverse,
                              order          = order,
                              preserve_range = True,
                              cval           = np.median(image))
    return image_tf

def pad_image(image,pad):
    new_image = np.pad(image,
                       ((pad,pad),(pad,pad),(0,0)),
                       mode="median")
    return new_image

def align_images(IMOBJCT, DYX, out_path, frame_mode="pad", order=5):

    #pad        = (np.abs(DYX).max(0)).astype(int)
    pad        = int(np.abs(DYX).max())
    new_images = []
    for i in tqdm(range(IMOBJCT[2])):

        frame = get_frame(IMOBJCT, i)

        if frame_mode=="pad":
            frame = pad_image(frame,pad)

        elif frame_mode=="crop":
            frame = frame[pad:-pad,pad:-pad]
        else:
            pass

        frame = apply_trafo(frame, DYX[i], order=order)

        cv2.imwrite(out_path+"images/"+str(10000001+i)[1:]+".png",frame)


if __name__ == '__main__':

    dev        = "cpu" ## "cuda:0"
    iters      = 11
    order      = 5
    mode       = None #"crop","pad"
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
    DYX  = get_trafos_seriel(IMOBJCT,model,dev=dev,iters=iters)
    #### cumsum to align al images in relation to the first frame
    DYXs = np.cumsum(DYX,0)
    #### subtract the median of x and y for centering the drift corrected
    DYXm = DYX-np.median(DYXs,0)

    #print(DYX)

    np.savetxt(out_path+"dyx.txt", DYX)
    np.savetxt(out_path+"dyx_cumsum.txt", DYXs)
    np.savetxt(out_path+"dyx_median.txt", DYXm)

    print("Apply transformation")
    align_images(IMOBJCT, -DYXm, out_path, frame_mode=mode, order=order)



            #fps     = IMOBJCT.get(cv2.CAP_PROP_FPS)
