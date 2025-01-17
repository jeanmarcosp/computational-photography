# -*- coding: utf-8 -*-
"""HW1_Jeanmarcos.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OaNnQhXBe5yJx29sWmsNW0IiV3-Fq8ZQ
"""

from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import os
import warnings
warnings.filterwarnings('ignore')

# Python initials

image_path = '/content/bonus2_raw.tiff'
image = io.imread(image_path)

height, width = image.shape

print("the image's dimensions are", image.shape)
print("the image has", image.dtype.itemsize * 8, "bits")

image_dp_array = image.astype(np.float64)

# Linearization

black = 2042
white = 16383

image_shifted = image_dp_array - black
image_scaled = image_shifted / (white - black)
image_clipped = np.clip(image_scaled, 0, 1)

# White balancing

r_channel = image_clipped[0::2, 0::2]
g_channel = (image_clipped[0::2, 1::2] + image_clipped[1::2, 0::2])/2
b_channel = image_clipped[1::2, 1::2]

def white_world(img):
    max_red = np.max(img[0::2, 0::2])
    max_blue = np.max(img[1::2, 1::2])

    green1 = np.max(img[0::2, 1::2])
    green2 = np.max(img[1::2, 0::2])
    max_green = np.mean([green1, green2])

    r_balanced = r_channel * (max_green/max_red)
    g_balanced = g_channel
    b_balanced = b_channel * (max_green/max_blue)

    white_world = np.array([r_balanced,
                            g_balanced,
                            b_balanced])

    return white_world

def gray_world(img):
    avg_red = np.mean(img[0::2, 0::2])
    avg_blue = np.mean(img[1::2, 1::2])

    green1 = np.mean(img[0::2, 1::2])
    green2 = np.mean(img[1::2, 0::2])
    avg_green = np.mean([green1, green2])

    r_balanced = r_channel * (avg_green/avg_red)
    g_balanced = g_channel
    b_balanced = b_channel * (avg_green/avg_blue)

    gray_world = np.array([r_balanced,
                            g_balanced,
                            b_balanced])

    return gray_world

def scaled_balancing():
    r_scale = 2.165039
    g_scale = 1.000000
    b_scale = 1.643555

    r_balanced = r_channel * r_scale
    g_balanced = g_channel * g_scale
    b_balanced = b_channel * b_scale

    scale_world = np.array([r_balanced,
                            g_balanced,
                            b_balanced])

    return scale_world

def manual_balancing(img):
    # patch 1
    blue_pixel = img[2877, 873]
    green_pixel = (img[2876,873] + img[2877,872])/2
    red_pixel = img[2876,872]

    # patch 2
    # blue_pixel = img[2931, 589]
    # green_pixel = (img[2930,589] + img[2931,588])/2
    # red_pixel = img[2930,588]

    # patch 3
    # blue_pixel = img[2269, 2425]
    # green_pixel = (img[2268,2425] + img[2269,2424])/2
    # red_pixel = img[2268,2424]


    r_balanced = r_channel * (green_pixel/red_pixel)
    g_balanced = g_channel * 1
    b_balanced = b_channel * (green_pixel/blue_pixel)

    manual_world = np.array([r_balanced,
                            g_balanced,
                            b_balanced])

    return manual_world

# white_balanced_image = scaled_balancing()
# white_balanced_image = white_world(image_clipped)
white_balanced_image = gray_world(image_clipped)
# white_balanced_image = manual_balancing(image_clipped)

# Bilinear Interpolation

all_mesh_grid_x, all_mesh_grid_y = np.arange(0, width), np.arange(0, height)

red_interp = scipy.interpolate.interp2d(np.arange(0, width, 2), np.arange(0, height, 2), white_balanced_image[0], kind='linear')

blue_interp = scipy.interpolate.interp2d(np.arange(1, width, 2), np.arange(1, height, 2), white_balanced_image[2], kind='linear')

green_interp1 = scipy.interpolate.interp2d(np.arange(0, width, 2), np.arange(1, height, 2), white_balanced_image[1], kind='linear')
green_interp2 = scipy.interpolate.interp2d(np.arange(1, width, 2), np.arange(0, height, 2), white_balanced_image[1], kind='linear')

green_interp = (green_interp1(all_mesh_grid_x, all_mesh_grid_y) + green_interp2(all_mesh_grid_x, all_mesh_grid_y))/2

interp_image = np.array([red_interp(all_mesh_grid_x, all_mesh_grid_y),
                         green_interp,
                        blue_interp(all_mesh_grid_x, all_mesh_grid_y)
                        ])

#Color Space Correction

srgb_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])

xyz_cam = np.array([[24542,-10860,-3401],
                   [-1490,11370,-297],
                   [2858,-605,3225]])

xyz_cam = xyz_cam / 10000.0

srgb_cam = np.dot(xyz_cam, srgb_xyz)
row_sums = srgb_cam.sum(axis=1)
normalized_srgb_cam = srgb_cam / row_sums[:, np.newaxis]

image_reshaped = np.dot(np.linalg.inv(normalized_srgb_cam), interp_image.reshape(3, -1))
color_corrected_image = image_reshaped.reshape(3, height, width)

# Brightness adjustment

grayscale = color.rgb2gray(color_corrected_image.transpose(1, 2, 0))
current_mean = np.mean(grayscale)

target_mean = 0.20

scaling_factor = target_mean / current_mean
brightened_image = color_corrected_image * scaling_factor
brightened_image_clipped = np.clip(brightened_image, 0, 1)

# Gamma encoding

nonlinear_transformed_image = np.where(brightened_image_clipped <= 0.0031308,
                                       12.92 * brightened_image_clipped,
                                       (1 + 0.055) * np.power(brightened_image_clipped, 1/2.4) - 0.055)

# PNG
plt.imsave('bonus2_code_developed.png', nonlinear_transformed_image.transpose(1, 2, 0))

# JPEG
# plt.imsave('patch3.jpg', nonlinear_transformed_image.transpose(1, 2, 0), pil_kwargs={'quality':95})