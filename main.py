#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pandas as pd
import os
import importlib
import time
import matplotlib.pyplot as plt
import ntpath
from sklearn.neighbors import NearestNeighbors
import datetime
import math
from tqdm import tqdm
import argparse


if not cv2.__version__ == '3.4.2':
    print('Warning: your opencv version is not 3.4.2, it might cause some error')

from utils import msop
from utils.stitch import read_meta, descspy2arr, feature_matching, pairwise_alignment, blending

# Settings
parser = argparse.ArgumentParser(description='Image Stitching')
parser.add_argument('--img-dir',default='./images/yard-001',  type=str,
                    help='path to image folder')
parser.add_argument('--meta-path',default = './images/yard-001/pano.txt', type=str,
                    help='path to meta data')
parser.add_argument('--save-dir',default = 'output', type=str,
                    help='dir for output image')                   
parser.add_argument('--nimgs', type=int, default=30,
                    help='the number of image you want to stitch')
parser.add_argument('--npys', type=int, default=2,
                    help='the number of layers of pyramid you want to calculate')
parser.add_argument('--rt', type=float, default=0.7,
                    help='Threshold for David Lowe’s ratio test when picking features. Lower is stricter.(default: 0.7)')
parser.add_argument('--vt', type=int, default=5,
                    help='Voting threshold in pixel(default: 5)')
parser.add_argument('--width-of-image', type=int, default=600,
                    help='the size of image, specified the width (default: 378)')




def getOutPath(SRCDIR):
    dt = datetime.datetime.now().strftime("-%H%M%S-%m%d")
    if(SRCDIR[-1]=='/'):
        SRCDIR = SRCDIR[:-1]
    return os.path.join('output', os.path.basename(SRCDIR)+dt+'.png')


def main():
    args = parser.parse_args()
    
    

    SRCDIR = args.img_dir 
    METAPATH = args.meta_path

    n_imgs = args.nimgs
    n_pys = args.npys

    voting_threshold = args.vt # in pixel
    rt_threshold = args.rt #Threshold for David Lowe’s ratio test when picking features. Lower is stricter.

    wsize = args.width_of_image
    imgsize = (wsize, int(wsize/4*3)) 
    
    # ## Dealing with Meta
    metas = read_meta(METAPATH)
    n_imgs = len(metas) if n_imgs > len(metas) else n_imgs


    ### Projection
    print('Computing Cylindrical Projection...')
    prjimgs = []
    orimgs = []
    fs = []
    for m in tqdm(metas[:n_imgs]):
        orimg = cv2.imread(os.path.join(SRCDIR, m['filename']))
        if(orimg.shape[0] > 400):
            orimg = cv2.resize(orimg, imgsize)
        prjimg= msop.cylindrical_projection(orimg, m['f'])
        prjimgs.append(prjimg)
        orimgs.append(orimg)
        fs.append(m['f'])


    ### Running MSOP

    importlib.reload(msop)
    print('Computing Features(MSOP)...')

    descspys = []
    imgpys = []
    for i in tqdm(range(n_imgs)):
        descspy, imgpy = msop.msop(orimgs[i], prj=True, focal_length=fs[i], n_pym = n_pys)
        descspys.append(descspy) # put original img into msop function
        imgpys.append(imgpy)



    ### Feature Matching

    print('Feature Matching...')
    pairsallpys = []
    for i in tqdm(range(len(descspys) - 1)):  
        a2bpairspy = feature_matching(descspys[i], descspys[i+1], rtThreshold = 0.7)
        pairsallpys.append(a2bpairspy)



    ### Determine Pairwise Alignment

    print('Running Alignment and Blending...')
    ms = []
    for i in range(len(pairsallpys)):
        bestm = pairwise_alignment(pairsallpys[i], descspys[i], descspys[i+1], n_pys = n_pys, ths = voting_threshold)
        ms.append(bestm)





    ### Blending

    cutvis = blending(ms, prjimgs)
    print('Saving Figure...')
    plt.figure(figsize = (100, 100))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(cutvis.astype('uint8'), cv2.COLOR_BGR2RGB))

    plt.savefig(getOutPath(SRCDIR), bbox_inches='tight', pad_inches=0)

    
if __name__ == '__main__':
    main()   


