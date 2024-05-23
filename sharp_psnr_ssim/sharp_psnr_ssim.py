# Oliver Huang, Jared Yoder
# Spring 2024
# Image Sharpening Algorithm for Approximate Multiplier benchmark
# Adapted from matlab code provided by the HPAM researchers

import cv2
import numpy as np
# from google.colab.patches import cv2_imshow
from skimage.metrics import structural_similarity as SSIM

import os
import math
import skimage
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.util import random_noise

lut = []
# lutname = 'LUT4.txt'
# lutname = 'LUT2.txt'
lutname = 'LUT2.txt'
with open(lutname) as f:
    for line in f:
        words = line.split()
        lut.append([int(x) for x in words])

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
print("Approximate Multiplier Used: %s" % lutname)

# noisy_img = skimage.img_as_ubyte(random_noise(img, mode='gaussian', seed=None, clip=True))

# Kernel
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
output_approx = np.zeros(I.shape, dtype=float)
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
            accumulator = 0
            ky, lx = Img_section.shape
            for k in range(ky):
                for l in range(lx):
                    # accumulator = accumulator + lut[Img_section[k][l]][Kernel[k][l]]
                    accumulator = accumulator + lut[Kernel[k][l]][Img_section[k][l]]
            output_accurate[i, j, color_channel] = 2 * I[i, j, color_channel] - temp_accumulate_acc / 273
            output_approx[i, j, color_channel] = 2 * I[i, j, color_channel] - accumulator / 273

        processed += ix
        # print("Processed Pixels: %d; %d%% Done" % (processed, math.ceil(100*(float(processed)/(x*y*3)))), end="\r")
        print("Processed Pixels: %d; %d%% Done" % (processed, int(100*(float(processed)/(ix*iy*icolor)))), end="\r")

# Unpadding (removing 2 rows & 2 colomns of zeros from each side)
output_acc = output_accurate[2:I.shape[0]-2, 2:I.shape[1]-2, :]
output_app = output_approx[2:I.shape[0]-2, 2:I.shape[1]-2, :]

# Convert to uint8, prevent clipping/peaking
output_acc = np.maximum(output_acc, np.zeros(output_acc.shape))
output_acc = np.minimum(output_acc, 255 * np.ones(output_acc.shape))
output_acc = output_acc.round().astype(np.uint8)
output_app = np.maximum(output_app, np.zeros(output_app.shape))
output_app = np.minimum(output_app, 255 * np.ones(output_app.shape))
output_app = output_app.round().astype(np.uint8)

# Display and save images
ogs = np.concatenate((img, output_acc), axis=1)
res = np.concatenate((ogs, output_app), axis=1)
# comparing original vs resized
print('\nOriginal, Exact Sharpen, Approx Sharpen')
cv2.namedWindow("results", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("results", 1920, 1080)
cv2.imshow("results", res)
# cv2.imshow("apprx", output_app)
# Resize the Window

# Calculate PSNR and SSIM
psnr = cv2.PSNR(output_acc, output_app)
print("PSNR:", psnr) #OpenCV prevents divide by 0 and has no 'Inf' so identical images will have PSNR of 361.20199909921956
ssim = SSIM(output_acc, output_app, data_range=output_app.max() - output_app.min(), multichannel=True)
#ssim = SSIM(output_acc, output_app, data_range=output_app.max() - output_app.min(), channel_axis=2) #newer versions
print("SSIM:", ssim)
print("\n[Ctrl+C to exit]")
cv2.imwrite('accurate_sharp.bmp', output_acc)
cv2.imwrite('approx_sharp.bmp', output_app)
while cv2.getWindowProperty('results', 0) >= 0:
    keyCode = cv2.waitKey(50)

cv2.destroyAllWindows()
# cv2.waitKey(1)
quit()
