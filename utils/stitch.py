import numpy as np
import cv2
import pandas as pd
import os
import importlib
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import math
import ntpath


def read_meta(METAPATH):
    """
        read meta data including focal length from meta file, which is generate from AutoStitch
    """
    with open(METAPATH) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 

    ii = 0
    metas = []
    func = None
    for line in content:
        if ii == 0:
            if func is None:
                func = ntpath.basename if '\\' in line else os.paht.basename
            meta = {'filename':func(line)}
        elif ii== 11:
            meta['f'] = float(line)
        elif ii == 12:
            ii = 0
            metas.append(meta)
            continue
        ii+=1
    return metas

def descspy2arr(descspy):
    """
        transform list of descriptor object into list of numpy array.
    """
    darrpy = []
    for descs in descspy:
        numd = len(descs)
        dim = descs[0].desc.flatten().shape[0]
        darr = np.zeros((numd, dim))
        for d in range(numd):
            darr[d, :] = descs[d].desc.flatten().astype('float64')
        darrpy.append(darr)
    return darrpy

def feature_matching(descpyA, descpyB, rtThreshold = 0.7):
    """
        rtThreshold: Threshold for David Lowe’s ratio test when picking features. Lower is stricter.
    """
    a2bpairspy = []
    for py in range(len(descpyA)):
        darrA = descspy2arr(descpyA)
        darrB = descspy2arr(descpyB)

        X = darrB[py] # 0 stage of pyramimd B
        Y = darrA[py]

        #print('X:', X)
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(Y)

        a2bpairs = []
        #print('indices:', indices)
        for ind in range(len(distances)):
            ds = distances[ind]
            # David Lowe’s ratio test
            if (ds[0] < ds[1] * rtThreshold):
                a2bpairs.append((ind, indices[ind][0]))
        
        a2bpairspy.append(a2bpairs)
    #print(a2bpairs)
    return a2bpairspy


def pairwise_alignment(pairspys, descspyA, descspyB, n_pys=1, ths = 5):
    """
        running pariwise alignment in descriptor A and descriptor B.
        
        @param
        
            pairspys: pairs of corresponding index.

            descspyA, descspyB: descriptors in pyramid.

            n_pys: number of layer of pyramid.

            ths: threshold for choosing 'fitting' features. 
    """
    coordspy = []
    for p in range(n_pys):
        descsA = descspyA[p]
        descsB = descspyB[p]
        a2bpairs = pairspys[p]
        npairs = len(a2bpairs)
        a2bcoords = np.zeros((npairs, 4))
           
        for i in range(npairs):
            aind, bind = a2bpairs[i]
            # b2acoords [i] = [a.x, a.y, b.x, b.y]       
            ax =  descsA[aind].point.x * (2**(2*p))
            ay =  descsA[aind].point.y * (2**(2*p))
            bx = descsB[bind].point.x * (2**(2*p))
            by = descsB[bind].point.y * (2**(2*p))
            a2bcoords[i] = [ax, ay, bx, by]         
        coordspy.append(a2bcoords)
        
    # RANSAC
    k = 60 # run RANSAC k times
    n = 2 # n samples are all inliers
    maxvote = 0
    
    npairs = len(pairspys[0]) # using stage 0 to calculate
    a2bcoords = coordspy[0]
    #print(a2bcoords.shape)
    
    for kdx in range(k):
        samcoords = a2bcoords[np.random.choice(npairs, n)]

        m1 = (1/n) * np.sum(samcoords[:, 0] -  samcoords[:, 2])
        m2 = (1/n) * np.sum(samcoords[:, 1] -  samcoords[:, 3])
        # dx = (m1 + b.x) - a.x      
        vote = 0
        for p in range(0, n_pys):
            pdx = np.abs(m1 +  coordspy[p][:, 2] - coordspy[p][:, 0]) 
            pdy = np.abs(m2 +  coordspy[p][:, 3] - coordspy[p][:, 1])
            vote += np.sum(((pdx < ths) & (pdy < ths))) 

        if vote > maxvote:
            maxvote = vote
            bestm = (m1, m2)
        
    return bestm
    
def blending(ms, prjimgs):
    """
       return blending images accordding to moving matrix ms(m1, m2).

    """
    dH = 0
    dW = 0
    for i in range(len(ms)):
        dH += math.ceil(math.fabs(ms[i][0])+1)*2
        dW += math.ceil(math.fabs(ms[i][1])+1)*2

    (hA, wA) = prjimgs[0].shape[:2]
    vis = np.zeros((hA +  dH, wA + dW, 3), dtype="float64")

    for idx in range( len(ms)):
        m1 ,m2 = ms[idx]

        (hA, wA) = prjimgs[idx].shape[:2]
        (hB, wB) = prjimgs[idx+1].shape[:2]

        blendh =  hA - int(math.fabs(m1))
        blendw =  wA - int(math.fabs(m2))

        mask11 = np.ones(prjimgs[idx].shape)
        mask21 = np.ones(prjimgs[idx].shape)
        mask12 = np.ones(prjimgs[idx].shape)
        mask22 = np.ones(prjimgs[idx].shape)



        absm2 = int(math.fabs(m2)) if int(math.fabs(m2)) > 0 else None
        absm1 = int(math.fabs(m1)) if int(math.fabs(m1)) > 0 else None
        mabsm2 = None if absm2 is None  else -absm2
        mabsm1 = None if absm1 is None  else -absm1

        w1 = math.fabs(m1)/(math.fabs(m2)+math.fabs(m1))
        w2 = math.fabs(m2)/(math.fabs(m2)+math.fabs(m1))

        m = mask11[absm1:, absm2:]
        mask11[absm1:, absm2:] = (m * np.arange((blendh), 0,  -1)[:, np.newaxis, np.newaxis] / blendh * w1)                                 + (m * np.arange((blendw), 0,  -1)[ np.newaxis, :, np.newaxis] / blendw * w2)
        m = mask21[:mabsm1, absm2:]
        mask21[:mabsm1, absm2:] = m * np.arange(0, (blendh), 1)[:, np.newaxis, np.newaxis] / blendh * w1                                 + m * np.arange((blendw), 0,  -1)[ np.newaxis, :, np.newaxis] / blendw * w2
        m = mask12[absm1:, :mabsm2]
        mask12[absm1:, :mabsm2] =  m * np.arange((blendh), 0,  -1)[:, np.newaxis, np.newaxis] / blendh * w1                                 + m * np.arange(0, (blendw), 1)[ np.newaxis, :, np.newaxis] / blendw * w2
        m =  mask22[:mabsm1, :mabsm2] 
        mask22[:mabsm1, :mabsm2] =  (m *  np.arange(0, (blendh), 1)[:, np.newaxis, np.newaxis] / blendh * w1)                                 + (m *  np.arange(0, (blendw), 1)[ np.newaxis, :, np.newaxis] / blendw * w2)


        if m2 >= 0 and m1 >= 0:
            amask = mask11
            bmask = mask22
        elif  m2 >= 0 and m1 < 0:
            amask = mask21
            bmask= mask12
        elif  m2 < 0 and m1 >=0 :
            amask = mask12
            bmask = mask21
        elif  m2 < 0 and m1 < 0:
            amask = mask22
            bmask = mask11

        ablended = prjimgs[idx] * amask
        bblended = prjimgs[idx+1] * bmask


        if (idx == 0):
            newOH = dH//2
            newOW = dW//2
            vis[newOH : newOH + hA, newOW : newOW + wA] = ablended # A

        else:
            vis[ bOH : bOH + hB, bOW : bOW + wB] = vis[ bOH : bOH + hB, bOW : bOW + wB]  * amask
            bblended = prjimgs[idx+1] * bmask
            newOH = bOH
            newOW = bOW
            
        bOH = math.ceil(newOH+m1)
        bOW = math.ceil(newOW+m2)
        vis[ bOH : bOH + hB, bOW : bOW + wB] += bblended # B


    tH = 0
    tW = 0
    minH = 1e8
    maxH = 0
    (hA, wA) = prjimgs[0].shape[:2]
    for i in range(len(ms)):
        tH += int(ms[i][0])
        if tH < minH:
            minH = tH
        if tH > maxH:
            maxH = tH
        tW += int(ms[i][1])
    if tW < 0:
        cutvis = vis[dH//2 + maxH : dH//2+hA+minH, dW//2+tW : dW//2+wA]
    else:
        cutvis = vis[dH//2 + maxH : dH//2+hA+minH, dW//2 : dW//2+wA+tW]
        
    return cutvis
        
      