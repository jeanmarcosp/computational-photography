#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
from hdr import (readImagesAndExposures, runRadiometricCalibration, mergeExposureStack, tonemap_photographic, XYZ2xyY,
                 xyY2XYZ, color_correction, gamma_correction, gray_world)
from cp_hw2 import lRGB2XYZ, XYZ2lRGB
from cp_exr import writeEXR

personal_image = False

if not personal_image:
    ### 1. HDR imaging
    ## Read images
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    dir_images = '../data/door_stack/'
    nExposure = 16
    scaling = 8  # debugging with small-size image
    option_input = 'rendered'  # ['rendered', 'RAW']

    img_list, exposure_list = readImagesAndExposures(dir_images, nExposure, option_input, scaling)

    ## Run radiometric calibration
    l = 5e1  # lambda
    option_weight = 'tent'  # ['uniform', 'tent', 'Gaussian', 'photon']
    g, w = runRadiometricCalibration(img_list, exposure_list, l, option_weight)
    # plt.plot(g)
    # plt.title('Camera response function (log)')
    # plt.savefig('personal-results/camera_response_function.png')
    # plt.show()

    ## Merge exposure stack into HDR image
    option_merge = 'logarithmic'  # ['linear', 'logarithmic']
    hdr = mergeExposureStack(img_list, exposure_list, g, w, option_input, option_merge, option_weight)

    ## Write hdr image (.exr format)
    brightness = 2 ** 6
    adjusted_hdr = (hdr / np.max(hdr)) * brightness

    # writeEXR("results/hdr_rendered_log_tent.exr", adjusted_hdr)
    print("wrote hdr image!")

    ### 2. Color correction and white balancing
    """ WRITE YOUR CODE HERE
    """
    color_corrected, hdr_color = color_correction(hdr)
    # writeEXR("results/hdr_color.exr", color_corrected)
    print("wrote color corrected image!")

    ### 3. Photographic tone mapping
    # tone map rgb
    K = 0.05
    B = 0.9

    tm_rgb = tonemap_photographic(hdr_color, K, B)
    # writeEXR("results/hdr_tonemap_rgb.exr", gamma_correction(tm_rgb))
    print("wrote rgb tone map image!")

    # tone map luminance only (xyY)
    np.seterr(divide='ignore', invalid='ignore')
    xyY = XYZ2xyY(lRGB2XYZ(hdr_color))
    xyY[:, :, 2] = tonemap_photographic(xyY[:, :, 2], K, B)
    tm_Y = XYZ2lRGB(xyY2XYZ(xyY))
    tm_Y = np.clip(tm_Y, 0, None)
    # writeEXR("results/hdr_tonemap_Y.exr", tm_Y)
    print("wrote luminance tone map image!")

else:
    ### 1. HDR imaging
    ## Read images
    if not os.path.isdir('./personal-results'):
        os.mkdir('./personal-results')
    dir_images = '../data/stair_stack/'
    nExposure = 16
    scaling = 4  # debugging with small-size image
    option_input = 'rendered'  # ['rendered', 'RAW']

    img_list, exposure_list = readImagesAndExposures(dir_images, nExposure, option_input, scaling)

    ## Run radiometric calibration
    l = 5e1  # lambda
    option_weight = 'tent'  # ['uniform', 'tent', 'Gaussian', 'photon']
    g, w = runRadiometricCalibration(img_list, exposure_list, l, option_weight)
    # plt.plot(g)
    # plt.title('Camera response function (log)')
    # plt.savefig('personal-results/camera_response_function.png')
    # plt.show()

    ## Merge exposure stack into HDR image
    option_merge = 'logarithmic'  # ['linear', 'logarithmic']
    hdr = mergeExposureStack(img_list, exposure_list, g, w, option_input, option_merge, option_weight)

    ## Write hdr image (.exr format)
    brightness = 2 ** 6
    adjusted_hdr = (hdr / np.max(hdr)) * brightness

    # writeEXR("personal-results/hdr_rendered_log_tent.exr", adjusted_hdr)
    print("wrote hdr image!")

    # personal tone map rgb
    K = 0.5
    B = 0.8

    hdr = gray_world(hdr)
    writeEXR("personal-results/hdr_gray_world.exr", gamma_correction(hdr))


    tm_rgb = tonemap_photographic(hdr, K, B)
    # writeEXR("personal-results/hdr_tonemap_rgb.exr", gamma_correction(tm_rgb))
    print("wrote rgb tone map image!")

    # personal tone map luminance only (xyY)
    np.seterr(divide='ignore', invalid='ignore')
    xyY = XYZ2xyY(lRGB2XYZ(hdr))
    xyY[:, :, 2] = tonemap_photographic(xyY[:, :, 2], K, B)
    tm_Y = XYZ2lRGB(xyY2XYZ(xyY))
    tm_Y = np.clip(tm_Y, 0, None)
    # writeEXR("personal-results/hdr_tonemap_Y.exr", tm_Y)
    print("wrote luminance tone map image!")
