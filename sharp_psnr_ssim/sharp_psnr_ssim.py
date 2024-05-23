# Oliver Huang, Jared Yoder
# EE 478/526, Spring 2024
# Image Sharpening Algorithm for Approximate Multiplier benchmark
# Adapted from matlab code provided by the HPAM researchers
# Sharpening method used: https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm 
# python3 ./sharp_psnr_ssim.py

import cv2 #pip3 install opencv-python==4.3.0.38 --user
import signal
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as SSIM #pip3 install scikit-image --user
# from google.colab.patches import cv2_imshow

import os
import math
import skimage
import argparse
from tqdm import tqdm
from skimage.util import random_noise 

signal.signal(signal.SIGINT, signal.SIG_DFL) #enable control c
lut4, lut2, lutW = [], [], []
lutname4, lutname2, lutnameW = 'LUT4.txt', 'LUT2.txt', 'LUTW.txt'
with open(lutname4) as f4:
    for line in f4:
        words = line.split()
        lut4.append([int(x) for x in words])

with open(lutname2) as f2:
    for line in f2:
        words = line.split()
        lut2.append([int(x) for x in words])

with open(lutnameW) as fW:
    for line in fW:
        words = line.split()
        lutW.append([int(x) for x in words])

# Read an image
# filename = '4.1.03.tiff' 
filename = '4103_crop.tiff'
# filename = '4103_crop.png'
# filename = 'jared.jpeg'
# filename = 'biden.jpg'
img = cv2.imread(filename)
processed = 0
print( "\
┌────────────────────────────────────────────────────┐\n\
│                                                    │\n\
│   ╔╗──────────╔═╦╗────────────╔╗                   │\n\
│   ╠╬══╦═╗╔═╦═╗║═╣╚╦═╗╔╦╦═╦═╦═╦╬╬═╦╦═╗              │\n\
│   ║║║║║╬╚╣╬║╩╣╠═║║║╬╚╣╔╣╬║╩╣║║║║║║║╬║  █ █ █   ▄▀  │\n\
│   ╚╩╩╩╩══╬╗╠═╝╚═╩╩╩══╩╝║╔╩═╩╩═╩╩╩═╬╗║  ▀▄▀▄▀ ▄▀    │\n\
│   ───────╚═╝───────────╚╝─────────╚═╝              │\n\
│   ********************************************     │\n\
│       ██╗░░██╗██████╗░░█████╗░███╗░░░███╗          │\n\
│       ██║░░██║██╔══██╗██╔══██╗████╗░████║          │\n\
│       ███████║██████╔╝███████║██╔████╔██║          │\n\
│       ██╔══██║██╔═══╝░██╔══██║██║╚██╔╝██║          │\n\
│       ██║░░██║██║░░░░░██║░░██║██║░╚═╝░██║          │\n\
│       ╚═╝░░╚═╝╚═╝░░░░░╚═╝░░╚═╝╚═╝░░░░░╚═╝          │\n\
│   ********************************************     │\n\
│                               by Oliver Huang      │\n\
│                                &  Jared Yoder      │\n\
└────────────────────────────────────────────────────┘")
print("Input image: %s" % filename)
# print("Approximate Multiplier Used: %s" % lutname)

# noisy_img = skimage.img_as_ubyte(random_noise(img, mode='gaussian', seed=None, clip=True))

# Gaussian Kernel with mean = 0 & sigma = 1
# reference : https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
Kernel = np.array([[1,  4,  7,  4, 1],
                   [4, 16, 26, 16, 4],
                   [7, 26, 41, 26, 7],
                   [4, 16, 26, 16, 4],
                   [1,  4,  7,  4, 1]])

# Pad the image with zeros # Oliver Note: this ends up causing white edges on the image. A flaw of their algorithm.
# I = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)
# Pad the image with edge values # Oliver Note: this is better
I = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='edge')
# Initialize output arrays
output_accurate = np.zeros(I.shape, dtype=float)
output_hpam4 = np.zeros(I.shape, dtype=float)
output_hpam2 = np.zeros(I.shape, dtype=float)
output_wallace = np.zeros(I.shape, dtype=float)

iy, ix, icolor = img.shape
y, x, color = I.shape

#  Convolution
for color_channel in range(color):
    for i in range(2, y - 2):
        for j in range(2, x - 2):

            # get a section of image from a current channel to multiply with
            # the kernel
            Img_section = I[i-2:i+3, j-2:j+3, color_channel]

            # ===================== ACCURATE COMPUTATION ===================
            # Element-by-element multiplication
            temp_multiply_acc = Img_section * Kernel

            # % Accumulation
            # temp_accumulate_acc = sum(temp_multiply_acc(:));
            temp_accumulate_acc = np.sum(temp_multiply_acc)

            # ==================== APPROXIMATE COMPUTATION =================
            # Element-by-element multiplication & accumulation
            accumulator4 = 0
            accumulator2 = 0
            accumulatorW = 0
            ky, lx = Img_section.shape

            for k in range(ky):
                for l in range(lx):
                    # paper access lut[Kernel][Img_section] (hypothesis); hpam4 clearly better than hpam2
                    # Comment other or this out
                    accumulator4 = accumulator4 + lut4[Kernel[k][l]][Img_section[k][l]]
                    accumulator2 = accumulator2 + lut2[Kernel[k][l]][Img_section[k][l]]
                    accumulatorW = accumulatorW + lutW[Kernel[k][l]][Img_section[k][l]]
                    # flipped access lut[Img_section] [Kernel] (different for HPAM!); hpam2 clearly better than hpam4
                    # reason for this is likely due to luck based on the Kernel which doesn't change
                    # accumulator4 = accumulator4 + lut4[Img_section[k][l]][Kernel[k][l]]
                    # accumulator2 = accumulator2 + lut2[Img_section[k][l]][Kernel[k][l]]
                    # accumulatorW = accumulatorW + lutW[Img_section[k][l]][Kernel[k][l]]
            output_accurate[i, j, color_channel] = 2 * I[i, j, color_channel] - temp_accumulate_acc / 273
            output_hpam4[i, j, color_channel] = 2 * I[i, j, color_channel] - accumulator4 / 273
            output_hpam2[i, j, color_channel] = 2 * I[i, j, color_channel] - accumulator2 / 273
            output_wallace[i, j, color_channel] = 2 * I[i, j, color_channel] - accumulatorW / 273

        processed += ix
        print("Processed Pixels: %d; %d%% Done" % (processed, int(100*(float(processed)/(ix*iy*icolor)))), end="\r")

# Unpadding (removing 2 rows & 2 colomns of zeros from each side)
output_acc = output_accurate[2:I.shape[0]-2, 2:I.shape[1]-2, :]
output_hp4 = output_hpam4[2:I.shape[0]-2, 2:I.shape[1]-2, :]
output_hp2 = output_hpam2[2:I.shape[0]-2, 2:I.shape[1]-2, :]
output_wal = output_wallace[2:I.shape[0]-2, 2:I.shape[1]-2, :]

# Convert to uint8, prevent clipping/peaking
output_acc = np.maximum(output_acc, np.zeros(output_acc.shape))
output_acc = np.minimum(output_acc, 255 * np.ones(output_acc.shape))
output_acc = output_acc.round().astype(np.uint8)

output_hp4 = np.maximum(output_hp4, np.zeros(output_hp4.shape))
output_hp4 = np.minimum(output_hp4, 255 * np.ones(output_hp4.shape))
output_hp4 = output_hp4.round().astype(np.uint8)

output_hp2 = np.maximum(output_hp2, np.zeros(output_hp2.shape))
output_hp2 = np.minimum(output_hp2, 255 * np.ones(output_hp2.shape))
output_hp2 = output_hp2.round().astype(np.uint8)

output_wal = np.maximum(output_wal, np.zeros(output_wal.shape))
output_wal = np.minimum(output_wal, 255 * np.ones(output_wal.shape))
output_wal = output_wal.round().astype(np.uint8)

# Calculate PSNR and SSIM
psnr4 = cv2.PSNR(output_acc, output_hp4)
psnr2 = cv2.PSNR(output_acc, output_hp2)
psnrW = cv2.PSNR(output_acc, output_wal)
ssim4 = SSIM(output_acc, output_hp4, data_range=output_hp4.max() - output_hp4.min(), multichannel=True)
ssim2 = SSIM(output_acc, output_hp2, data_range=output_hp2.max() - output_hp2.min(), multichannel=True)
ssimW = SSIM(output_acc, output_wal, data_range=output_wal.max() - output_wal.min(), multichannel=True)
#ssim = SSIM(output_acc, output_app, data_range=output_app.max() - output_app.min(), channel_axis=2) #newer versions
print("\n")
print("┌───────────────────────────────────┐")
print("|         Sharpening Report         |")
print("|-----------------------------------|")
print("|    Peak Signal to Noise Ratio     |")
print("|-----------------------------------|")
print("| PSNR   HPAM4:   %.14f" % psnr4,  "|") 
print("| PSNR   HPAM2:   %.14f" % psnr2,  "|")
print("| PSNR Wallace:  %.14f"  % psnrW,  "|") # OpenCV prevents divide by 0 and has no 'Inf', 
print("|-----------------------------------|") # so identical images will have PSNR of 361.20199909921956
print("|Structural Similarity Index Measure|") # reference: https://stackoverflow.com/a/61847143 
print("|-----------------------------------|")
print("| SSIM   HPAM4:  %.16f" % ssim4,   "|")
print("| SSIM   HPAM2:  %.16f" % ssim2,   "|")
print("| SSIM Wallace:  %.16f" % ssimW,   "|")
print("└───────────────────────────────────┘")
print("\n[Ctrl+C to exit]")
cv2.imwrite('accurate_sharp.bmp', output_acc)
cv2.imwrite('hpam4_sharp.bmp', output_hp4)
cv2.imwrite('hpam2_sharp.bmp', output_hp2)
cv2.imwrite('wallace_sharp.bmp', output_wal)

#subplot(r,c) provide the no. of rows and columns
fig, axarr = plt.subplots(2,2) 

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0,0].set_title("Original")
axarr[0,1].set_title("Wallace")
axarr[1,0].set_title("HPAM4")
axarr[1,1].set_title("HPAM2")
axarr[0,0].imshow(img[...,::-1]) #cv2 uses bgr, [...,::-1] convert from BGR to RGB
axarr[0,1].imshow(output_wal[...,::-1])
axarr[1,0].imshow(output_hp4[...,::-1])
axarr[1,1].imshow(output_hp2[...,::-1])

fig.suptitle("Image Sharpening")
plt.setp(axarr, xticks=[], yticks=[])
plt.show()
