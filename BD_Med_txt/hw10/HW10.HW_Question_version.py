# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# HW10
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import math
import random
from scipy.io import loadmat


# We need pydicom library to read DICOM
# conda install -c conda-forge pydicom
# https://medium.com/@hengloose/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed
import pydicom


# Load diferent images, add optional noise
def LoadImage(nImg=0, dNoise = 0, vmin=-1, vmax=-1):
    
    factor = 1.0
    
    
    # Experimenting with different images here; 
    # use nImg=0 for the "triangle" image discussed in the class
    if nImg==-1:
        Img = loadmat('./Data/FracturesNoisy.mat').get("Img")
    elif nImg==0:
        Img = Image.open('./Data/Shapes0.bmp')
        factor = 255 # this image has binary pixels, scale to 255
    elif nImg==1:
        Img = Image.open('./Data/Lenna.bmp')
    elif nImg==2:
        Img = Image.open('./Data/Circles.bmp')
    elif nImg==3:
        f = pydicom.dcmread('./Data/Large_Skull')
        Img = f.pixel_array
    else:
        f = pydicom.dcmread('./Data/test_CT');  
        Img = f.pixel_array
     
    # Make sure we use float pixel values for precision
    Img = np.array(Img)
    Img = factor*Img.astype(float)
        
    # Add uniform noise if needed
    if dNoise>0:
        np.random.seed(0) # seed for reproducible results
        Img = Img + np.random.uniform(-dNoise, dNoise, np.shape(Img))
        #Img = Img + np.random.normal(0, dNoise, np.shape(Img))
        
    # Display the image 
    plt.figure("Loaded image")
    if vmin>vmax:
        plt.imshow(Img, cmap=plt.cm.bone, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(Img, cmap=plt.cm.bone)
    
    # Return loaded image
    return Img




# Compute LoG filter weights
def LoG_Weights(sigma, radFilter=3):
    # Initialize filter and its parameters
    r = radFilter # filter radius R, you can experiment with this
    W = np.zeros((2*r+1, 2*r+1))
    s2 = 2*sigma**2 # to avoid computing this every time
    d=r+1
    for x in range(2*r+1):
        x1 = x-d;
        for y in range(2*r+1):
            y1 = y-d
            t=(x1**2+y1**2)/s2
            W[x,y]=(1-t)*math.exp(-t)
        
    #F = F/sum(sum(F)); % normalize to unit sum – why?
    W = W/sum(W) # normalize to unit sum to keep the same intensity average
    return W;
    
    
    
    
# Denoising function
def ImageFilter(Img, nFilter, radFilter=3):
    # Get image sizes
    nRows, nCols = np.shape(Img)
    
    # Initialize Filtered image as zeros
    Filtered = np.zeros((nRows, nCols))
    
    # Filter radius - just a shorter variable name
    r = radFilter
    WLoG = LoG_Weights(6, r)
    
    # Filter
    for nr in range(r+1, nRows-r-2):
        for nc in range(r+1, nCols-r-2):
            # Filter at pixel [nr, nc]
            if nFilter==2: #LoG filter
                Filtered[nr][nc] = np.sum(Img[nr-r:nr+r+1, nc-r:nc+r+1]*WLoG) 
            elif nFilter==3: # Gradient
                px = Img[nr+1,nc] - Img[nr-1,nc] # intensity change along x
                py = Img[nr,nc+1] - Img[nr,nc-1] # intensity change along y
                Filtered[nr][nc] = np.sqrt(px**2 + py**2)    
            else: # Average filter by default
                Filtered[nr][nc] = np.mean(Img[nr-r:nr+r, nc-r:nc+r]) 
            
            
    # Display original and denoised images
    plt.figure("Filtered image")
    plt.imshow(Filtered, cmap=plt.cm.bone)
    
    # Return
    return Filtered




# Function to find lines in the edge image EdgeImg
def FindLine(EdgeImg):

    # Find image sizes
    nrows, ncols = np.shape(EdgeImg)
    
    # Convert the image to 1D buffer, to compute quantiles
    t = max(0,np.percentile(EdgeImg, 99))  # select edge intensity threshold
    print('\n T = ', t)
    
    # Set minimal and maximal line size
    r0 = 100     # min line size
    r1 = 2*r0   # max line size
    
    # Search for lines
    Lines = [] #initialize the list of detected lines
    for x0 in range(nrows):
        if x0 % round(nrows/10)==0:
            print('\n Progress: ', round(100*x0/nrows)) # display current progress
        for y0 in range(ncols):
            
            
            if EdgeImg[x0,y0] < t:
                continue 
            
            for x1 in range(x0+1, min(x0+r1,nrows)):
                for y1 in range(max(1,y0-r1),min(ncols,y0+r1)):
                    
                    
                    if EdgeImg[x1,y1] < t:
                        continue 
                    
                   
                    # if EdgeImg[(x0+x1)//2, (y0+y1)//2] < t:
                    #     continue 
                    
                    
                    d = math.sqrt( (x1-x0)**2+(y1-y0)**2 );
                    if d<r0 or d>r1:
                        continue 
                        
                    # Compute line slope and intercept
                    a = (y1-y0)/(x1-x0)
                    b = y0-a*x0
                    
                    # Compute line cost function
                    C=0
                    nSamples = 50
                    dx=(x1-x0)/nSamples
                    for ns in range(nSamples+1):
                        x = x0+ns*dx
                        C = C+EdgeImg[round(x), round(a*x+b)]
                    
                   
                    Lines.append( (C, x0, y0, x1, y1, d) )
                    
    
    # Sort detected lines by strength
    Lines.sort(key = lambda x: -x[0])  
    #nLines = len(Lines)//600 # edges with the highest cost
    nLines = 10
    print('\n Selected line count:', nLines)
    
    # Draw the best lines
    E = np.zeros((nrows, ncols))
    for nL in range(nLines):
        C, x0, y0, x1, y1, d = Lines[nL]
        a = (y1-y0)/(x1-x0)
        b = y0-a*x0
        dx=(x1-x0)/d # Making enough steps here to make this line visible!
        for x in np.arange(x0,x1,dx):
             E[int(round(x)), int(round(a*x+b))] = 255;
             
             
    # Display the image 
    plt.figure("Loaded image")
    Emin = np.percentile(E, 50)
    Emax = np.percentile(E, 100)
    print('\n Emin, Emax =  ', Emin, Emax)
    #E = 200*(E-Emin)/(Emax-Emin)
    plt.imshow(E, cmap=plt.cm.bone, vmin=Emin, vmax=Emax)
    
    return E

# Just testing some filters here
bEdgeilter = False
if bEdgeilter:
    Img = LoadImage(1)
    Filtered = ImageFilter(Img, 2, 10)
    

############################################################
# 
#                   Main part
#
############################################################

          
Img = LoadImage(0, 400)

EdgeImg = ImageFilter(Img, 3)

FindLine(EdgeImg)

