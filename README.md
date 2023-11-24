# Fiduciary-free frame alignment and image stabilization

Command-line tool and supplementary material to the paper: "Fiduciary-free frame alignment for robust time-lapse drift correction estimation in multi-sample cell microscopy"

![overview](images/idea.svg)

This approach for jitter and drift correction is based on ["RAFT: Recurrent All Pairs Field Transforms for Optical Flow"](https://arxiv.org/pdf/2003.12039.pdf)by Zachary Teed and Jia Deng (ECCV 2020).

## Install

- **requirements:**
-- python 3.6+ \
   `pip3 install gdown torch torchvision torchaudio opencv-contrib-python` \
   `pip3 install scipy tqdm path imageio scikit-image joblib`
-- `git clone https://github.com/StefanBaar/cell_align`
-- `cd cell_align`

## Running
   ```bash
   python3 stabilize.py [input path to image dic or video]
   ```
## Samples

- DMSO (left: raw, right stabilized)

[![DMSO](https://img.youtube.com/vi/gazuq-znHJ4/hqdefault.jpg)](https://youtu.be/gazuq-znHJ4)
- RA (left: raw, right stabilized)

[![RA](https://img.youtube.com/vi/PBX6gSWabdU/hqdefault.jpg)](https://youtu.be/PBX6gSWabdU)
- KNK808 (left: raw, right stabilized)

[![KNK808](https://img.youtube.com/vi/OyPupI3irXw/hqdefault.jpg)](https://youtu.be/OyPupI3irXw)




## TODO:
- Model: replace old sintel model url


<!--- This repo requires RAFT
git submodule add https://github.com/princeton-vl/RAFT -->
