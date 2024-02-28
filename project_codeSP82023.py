# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 23:25:50 2023

@author: Jeremy Stockdill
"""


import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import functools
import glob
from PIL import Image

#PMMA microspheres
mu = 0.6782*100 # (cm^-1 -> m^-1) [20keV: 0.6782 | 24keV: 0.4895]
delta = 6.61E-07 #ts-imaging database [20keV: 6.61E-07 | 24keV: 4.59E-07]
px_size = 10.12E-6 #m 
r2 = 0.2 # (m)
mag = 1.
# (m^-1)

flats1 = glob.glob('C:/Users/bartn/OneDrive/Pictures/2023Sp820_20cm_flats/*.tif')
flats = np.array([np.array(Image.open(fname)) for fname in flats1])
flatsarray = (np.sum(flats, axis=0))/len(flats)

deg0 = glob.glob('C:/Users/bartn/OneDrive/Pictures/2023Sp820_20cm_deg0/*.tif')
deg01 = np.array([np.array(Image.open(fname)) for fname in deg0])
deg0array = (np.sum(deg01, axis=0))/len(deg01)

Izarray = deg0array/flatsarray
Izarray = Izarray[30:]

I = np.vstack((Izarray[::-1], Izarray)) #stacking to avoid phase wrapping
I = I/0.985 #correction to make I=1 outside objects

from matplotlib import pyplot as plt
plt.imshow(I, cmap='gray', interpolation='nearest')
plt.show()

im = Image.fromarray(I)
im.save("C:/Users/bartn/OneDrive/Desktop/python_files/Intensity23.tif")

#%%
@functools.lru_cache()
def kspace(px_size, shape):
        lim = np.pi / px_size
        rows, cols = shape
        kperp2 = np.hypot(*np.ogrid[-lim:lim:1j*rows, -lim:lim:1j*cols])**2
        kperp2 = fftshift(kperp2)
        return kperp2
        

def pbi_sipr_kitchen(im, mu, delta, px_size, r2, mag, mu2=None, delta2=None):
        """ Perform TIE (Paganin) phase retrieval on a flat-fielded image to determine
        intensity from a propagation-based phase-contrast image (PBI).

        Can be called *either* with one or two materials mu and delta.

        Note: This code differs from David Paganin's 2002 notation with magnification
        Here we only divide the propagation distance by M, but we assume the 
        effective pixel size has been measured in the image plane.

        Args:
            im: 2d ndarray image after dark and flat-field correction
            mu: linear attenuation coefft in m^-1
            delta: real part of the refractive index increment
            px_size: effective pixel size in metres with magnification taken into account, with  e.g. 5E-6
            r2: object to detector propagation distance (m)
            mag: magnification due to point source
            (optionally provide mu2, delta2):
                mu2: linear attenuation coefft of encasing material in m^-1
                delta2: real part of the refractive index increment of encasing material
                    
        Returns:
            phase-retrieved thickness map in m

        """

        # check whether we have everything required for a second material
        material2_provided = mu2 is not None and delta2 is not None
        if material2_provided:
            mu = mu2 - mu
            delta = delta2 - delta
            
        kperp2=kspace(px_size, im.shape)
        numerator = mu * fft2(im)
        #Note: The division by magnification below accounts for the 
        #Fresnel scaling theorem that reduces phase contrast by factor M.
        denominator = r2 * delta * kperp2/mag + mu
        intensity = (ifft2(numerator / denominator)).real

        return intensity

PBIdata = pbi_sipr_kitchen(I, mu, delta, px_size, r2, mag, mu2=None, delta2=None)
Thickness = -1*(np.log(PBIdata))/mu #From attenuation equation I = exp(-mu*t)
#im = Image.fromarray(Thickness)
#im.save("Thicknessunc.tif")
Thickness = np.where(Thickness < 9e-4, 0, Thickness) #values below 0.0009 -> 0
plt.imshow(Thickness, cmap='gray', interpolation='nearest')
plt.show() # make small values constant, use PBIdatasmooth for PBIlaplace

im = Image.fromarray(Thickness)
im.save("C:/Users/bartn/OneDrive/Desktop/python_files/Thickness23.tif")
#%%
#Thickness =  Image.open("C:/Users/bartn/OneDrive/Desktop/python_files/Thickness23.tif")
#Thickness = np.array(Thickness)
plt.imshow(Thickness, cmap='gray', interpolation='nearest')
plt.show()

gradT1 = np.gradient(Thickness,px_size,axis=0)
gradT1 = np.gradient(gradT1,px_size,axis=0)
gradT2 = np.gradient(Thickness,px_size,axis=1)
gradT2 = np.gradient(gradT2,px_size,axis=1)
LaplaceT = gradT1 + gradT2
avL1 = np.average(LaplaceT)
plt.imshow(LaplaceT, cmap='gray', interpolation='nearest')
plt.show()
im = Image.fromarray(LaplaceT)
im.save("C:/Users/bartn/OneDrive/Desktop/python_files/LaplacianT23.tif")

gradT = np.gradient(Thickness,px_size,axis=None)
gradT = (gradT[0])**2 + (gradT[1])**2
plt.imshow(gradT, cmap='gray', interpolation='nearest')
plt.show()
im = Image.fromarray(gradT)
im.save("C:/Users/bartn/OneDrive/Desktop/python_files/Gradient23.tif")

etanum1 = I - np.exp(-mu*Thickness)*(1 + r2*delta*LaplaceT)
etaden1 = np.exp(-mu*Thickness)*(r2**2)*LaplaceT
#%%
etanum2 = I - np.exp(-mu*Thickness)*(1 - r2*delta*(mu*gradT - LaplaceT))
avnum = np.average(etanum2, axis=None)
etaden2 =  (r2**2)*np.exp(-mu*Thickness)*(LaplaceT*(1 - mu*Thickness) - mu*(gradT)*(2-mu*Thickness))
avden = np.average(etaden2, axis=None)

avcheck = LaplaceT*(1 - mu*Thickness)
avck = np.average(avcheck, axis=None)

#%%
#eta = etanum1/etaden1
eta = etanum2/etaden2
eta = np.where(Thickness < 9e-4, 0, eta)

plt.imshow(eta, cmap='gray') #, interpolation='nearest')
plt.show()

im = Image.fromarray(eta)
im.save("C:/Users/bartn/OneDrive/Desktop/python_files/etavalues23_20keV_20cm.tif")


