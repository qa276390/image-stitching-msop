#msop
import numpy as np
import cv2
import pandas as pd
import os
import importlib
import time
import matplotlib.pyplot as plt
import os
from skimage.exposure import rescale_intensity
import copy
import math
from PIL import Image
from scipy.ndimage import filters




class point2d():
    def __init__(self, value, x=-1, y=-1, coord=None):
        if coord is not None:
            self.x = coord[0]
            self.y = coord[1]
        else:
            self.x = x 
            self.y = y
            
        self.value = value


class descriptor():
    def __init__(self, point, ori, desc):
        self.orientation = ori
        self.point = point
        self.desc = desc
    def setdesc(self, desc):
        self.desc = desc



def compute_harris_responce(im, sigma=1.5):
    """
        Compute the Harris response of each pixel
        in the gray-scale image.
    """
    from scipy.ndimage import filters
    import numpy as np

    Ix = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), Ix)
    Iy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), Iy)
    
    # compute Harris corner
    Wxx = filters.gaussian_filter(Ix**2, sigma)
    Wxy = filters.gaussian_filter(Ix*Iy, sigma)
    Wyy = filters.gaussian_filter(Iy**2, sigma)

    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    return Wdet/(Wtr+1e-8)

  
    

def upscaling(src, idx):
    if idx <= 0:
        return src 
    src = upscaling(cv2.pyrUp(cv2.pyrUp(src)), idx-1)
    return src


def masksurby(src, ind, r, num):
    # for grayscale
    L0 = (ind[0] - r) if ((ind[0] - r) >= 0) else 0
    L1 = (ind[1] - r) if ((ind[1] - r) >= 0) else 0
    U0 = (ind[0] + r) if ((ind[0] + r) <= src.shape[0]) else src.shape[0]
    U1 = (ind[1] + r) if ((ind[1] + r) <= src.shape[1]) else src.shape[1]
    src[L0:U0, L1:U1] = num
    return src
def maskonly(src, ind, diameter):
    r = diameter//2
    masked = np.zeros((r, r))
    max0, max1= src.shape
    # for grayscale
    L0 = (ind[0] - r) if ((ind[0] - r) >= 0) else 0
    L1 = (ind[1] - r) if ((ind[1] - r) >= 0) else 0
    U0 = (ind[0] + r) if ((ind[0] + r) <= max0) else max0
    U1 = (ind[1] + r) if ((ind[1] + r) <= max1) else max1
    masked = src[L0:U0, L1:U1]
    return masked


def nonmaxsup(src, maxf=500, r=-1):
    # while (not enough feature point selected)
    # set r - 1
    # while (features is not all zeros)
    #   sort features
    #   find max(selected), and replace its surrounded and itself by zeros
    fps = []
    if r == -1:
        r = src.shape[0]//10
        
    while(len(fps) <= maxf):
        r = r - 1
        fmap = copy.deepcopy(src)
        
        # mask sur
        for p in fps:
            fmap =  masksurby(fmap, (p.x, p.y), r, 0)
        maxn = 1e8
        while(maxn > src.max()*0.05):
            ind = np.unravel_index(np.argmax(fmap, axis=None), fmap.shape)
            maxn = fmap[ind]
            fmap = masksurby(fmap, ind, r, 0)
            
            fps.append(point2d(value = maxn, coord = ind)) # collect new selected feature
    return fps

def cylindrical_projection(img, focal_length):
    if(len(img.shape)>2):
        height, width, _ = img.shape
    else:
        height, width = img.shape
    cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)
    
    for y in range(-int(height/2), int(height/2)):
        for x in range(-int(width/2), int(width/2)):
            cylinder_x = focal_length*math.atan(x/focal_length)
            cylinder_y = focal_length*y/math.sqrt(x**2+focal_length**2)
            
            cylinder_x = round(cylinder_x + width/2)
            cylinder_y = round(cylinder_y + height/2)

            if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                cylinder_proj[cylinder_y][cylinder_x] = img[y+int(height/2)][x+int(width/2)]
    
    # Crop black border
    # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    if(len(img.shape)>2):
        _, thresh = cv2.threshold(cv2.cvtColor(cylinder_proj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(cylinder_proj, 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
        
    return cylinder_proj[y:y+h, x:x+w]



def msop(orimg, maxfeat=300, n_pym = 2, prj=False, focal_length=0):
    img = cv2.cvtColor(orimg,cv2.COLOR_BGR2GRAY)
    
    ### Harris Corner Response
    print('Computing Harris Corner Response...')
    hrpy = [] #harris response pyramid
    srcs = [] #image source pyramid
    src = img
    for i in range(n_pym):
        dst = compute_harris_responce(src)
        #srcs.append(cv2.resize(upscaling(src, i), (img.shape[1], img.shape[0]))) # store the origin size src
        #hrpy.append(cv2.resize(upscaling(dst, i), (img.shape[1], img.shape[0]))) # store the origin size response
        srcs.append(src) # store the origin size src
        hrpy.append(dst) # store the origin size response
        src = cv2.pyrDown(cv2.pyrDown(src))
        
    ### Projection to Cylindrical 
    print('Projection to Cylindrical...')
    if (prj):
        if(focal_length==0):
            print('Please Specify Focal Length!')
            return None
        for i in range(n_pym):
            hrpy[i] =  cylindrical_projection(hrpy[i], focal_length)
            srcs[i] =  cylindrical_projection(srcs[i], focal_length)
        
    ### Non Maximum Suppression
    print('Computing Non Maximum Suppression...')
    fpspy = []
    for hr in hrpy:
        fpspy.append(nonmaxsup(hr, maxf=maxfeat))

    ### Descriptor (Orientation included)
    print('Constructing Descriptor...')
    descspy = []
    for i in range(len(fpspy)):
        fps = fpspy[i]
        src = srcs[i]
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=5)
        descs = []
        for fp in fps:
            ind = (fp.x, fp.y)
            rmat = cv2.getRotationMatrix2D(ind, math.atan2(gx[ind], gy[ind]), scale=1.0)
            rimg = cv2.warpAffine(src, rmat, (src.shape[1], src.shape[0]))
            masked = maskonly(rimg, ind, 40)
            if (masked.shape[1]==0):
                print('error in desc ind:', ind)
                break
            blurf = cv2.GaussianBlur(masked, ksize=(5, 5), sigmaX=1, sigmaY=1)
            resizef = cv2.resize(blurf, (8, 8)) # 8*8
            normf = (resizef-np.mean(resizef))/(np.std(resizef) + 1e-8)
            descs.append(descriptor(fp, (gx[ind], gy[ind]), normf))
        descspy.append(descs)
        
        return descspy

