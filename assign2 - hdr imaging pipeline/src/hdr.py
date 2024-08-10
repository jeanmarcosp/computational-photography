#!/usr/bin/python
import os

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from cp_hw2 import read_colorchecker_gm
from skimage.transform import resize


def readImagesAndExposures(dir_images, nExposure, option_input, scaling=1):
    """ Read image and exposures """
    print('Reading images... ')

    exposure_list = []
    img_list = []

    for i in range(1, nExposure + 1):
        if option_input == 'rendered':
            filename = dir_images + "exposure" + str(i) + '.jpg'
            im = io.imread(filename)
            im = im[::scaling, ::scaling]
            im = im.astype(np.float32)
            img_list.append(im)
        elif option_input == 'RAW':
            filename = dir_images + "exposure" + str(i) + '.tiff'
            im = io.imread(filename)
            im = im[::scaling, ::scaling]
            im = im.astype(np.float32)
            img_list.append(im)

        exposure_list.append((1 / 2048) * pow(2, i - 1))

    print('Done!')

    return img_list, exposure_list


def runRadiometricCalibration(img_list, exposure_list, l, option_weight, scaling=200):
    """ Run radiometric calibration
    """
    image_array = np.asarray(img_list)[::, ::scaling, ::scaling, ::]
    image_array = image_array.reshape(image_array.shape[0], image_array.shape[1] * image_array.shape[2] * image_array.shape[3]).transpose()

    weights = np.linspace(0, 1, 256)
    clipped_weights = weights[(0.05 <= weights) & (weights <= 0.95)]

    w = np.zeros_like(weights)
    valid = np.where((0.05 <= weights) & (weights <= 0.95))[0]

    if option_weight == 'uniform' or option_weight == 'photon':
        w[valid] = 1

    elif option_weight == 'Gaussian':
        w[valid] = np.exp((-1) * (pow(clipped_weights - 0.5, 2)))

    elif option_weight == 'tent':
        w[valid] = np.where(clipped_weights < (1 - clipped_weights), clipped_weights, (1 - clipped_weights))

    log_exposure = np.log(np.asarray(exposure_list))
    g, log_L = gsolve(image_array * 255, log_exposure, l, w, option_weight)

    return g, w


def mergeExposureStack(img_list, exposure_list, g, w, option_input, option_merge, option_weight):
    """ Merge exposure stack into HDR image
    """
    exposure_array = np.asarray(exposure_list)

    image_array = np.asarray(img_list).transpose((1, 2, 3, 0))
    image_array *= pow(2, 8) - 1
    image_array = image_array.astype(np.uint8)
    image_array = np.where(image_array > 255 * 0.95, 255, image_array)
    image_array = np.where(image_array < 255 * 0.05, 0, image_array)

    linear_image = None
    if option_input == 'rendered':
        linear_image = np.exp(g[image_array])

    elif option_input == 'RAW':
        linear_image = np.asarray(img_list).transpose((1, 2, 3, 0)) / (pow(2, 16) - 1)

    hdr = None
    if option_merge == 'logarithmic' and option_weight == 'photon':
        term1 = np.sum((w[image_array] * exposure_array[None, None, None, :] * (np.log(linear_image + 0.00001) - np.log(exposure_array[None, None, None, :]))), axis=3)
        term2 = np.sum(w[image_array] * exposure_array[None, None, None, :] + 0.001, axis=3)
        hdr = np.exp(np.nan_to_num(np.divide(term1, term2)))
    elif option_merge == 'logarithmic':
        term1 = np.sum((w[image_array] * (np.log(linear_image + 0.001) - np.log(exposure_array[None, None, None, :]))), axis=3)
        term2 = np.sum(w[image_array], axis=3)
        hdr = np.exp(np.nan_to_num(np.divide(term1, term2)))

    if option_merge == 'linear' and option_weight == 'photon':
        term1 = np.sum(w[image_array] * linear_image, axis=3)
        term2 = np.sum(w[image_array] * exposure_array[None, None, None, :] + 0.001, axis=3)
        hdr = np.nan_to_num(np.divide(term1, term2))
    elif option_merge == 'linear':
        term1 = np.sum((w[image_array] * linear_image / exposure_array[None, None, None, :]), axis=3)
        term2 = np.sum(w[image_array] + 0.00001, axis=3)
        hdr = np.nan_to_num(np.divide(term1, term2))

    return hdr


def gsolve(I, log_t, l, w, option_weight):
    """ Solve for imaging system response function

    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system response function g as well as the log film irradiance
    values for the observed pixels.

    This code is from the following paper:
    P. E. Debevec and J. Malik, Recovering High Dynamic Range Radiance Maps from Photographs, ACM SIGGRAPH, 1997

    Parameters
    ----------
    I(i, j): pixel values of pixel location number i in image j (nPixel, nExposure)
    log_t(j): log delta t, or log shutter speed for image j (nExposure)
    l: lambda, the constant that determines the amount of smoothness
    w(z): weighting function value for pixel value z (256)

    Returns
    -------
    g(z): the log exposure corresponding to pixel value z
    log_L(i) is the log film irradiance at pixel location i
    """

    n = 256
    nPixel = I.shape[0]
    nExposure = I.shape[1]

    A = np.zeros((nPixel * nExposure + n + 1, n + nPixel))
    b = np.zeros((A.shape[0],))

    # Include the data-fitting equations
    k = 0
    for i in range(nPixel):
        for j in range(nExposure):
            z = I[i, j].astype(np.uint8)

            if option_weight == 'photon':
                wij = np.exp(log_t[j])
            else:
                wij = w[z]

            A[k, z] = wij
            A[k, n + i] = -wij
            b[k] = wij * log_t[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1  # b[k] = 0
    k += 1

    # Include the smoothness equations
    for z in np.arange(1, n):
        A[k, z - 1] = l * w[z]
        A[k, z] = -2 * l * w[z]
        A[k, z + 1] = l * w[z]
        k += 1

    print(A.shape)
    # exit()
    # Solve the system
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    g = x[:n]
    log_L = x[n:]

    return g, log_L


def tonemap_photographic(imIn, key, burn):
    """ Implementation of Reinhard et al., Photographic Tone Reproduction for
    Digital Images, SIGGRAPH 2002.
    """

    """ WRITE YOUR CODE HERE
    """
    im_m = np.exp((1 / (imIn.shape[0] * imIn.shape[1])) * np.sum(np.log(imIn + 0.001)))
    im_t = imIn * (key / im_m)

    return (im_t * (1 + (im_t / (pow(burn * np.max(im_t), 2))))) / (1 + im_t)


def gamma_correction(img_in):
    """ WRITE YOUR CODE HERE
    """
    img_in *= (0.20 / np.mean(color.rgb2gray(img_in)))
    img_clip = np.clip(img_in, 0, 1)

    return np.where(img_clip <= 0.0031308, (12.92 * img_clip), (((1 + 0.055) * (img_clip ** (1 / 2.4))) - 0.055))


def color_correction(img_in):
    # plt.imshow(gamma_correction(img_in))
    # color_clicks = np.asarray(plt.ginput(24, 120))
    # plt.close()

    color_patches = np.array(
        [[421, 185], [421, 170], [421, 146], [421, 131], [421, 105], [421, 91], [439, 189], [439, 166], [439, 151],
         [439, 125], [439, 109], [439, 85], [459, 189], [459, 166], [459, 149], [459, 128], [459, 109], [459, 88],
         [481, 185], [481, 170], [481, 150], [481, 130], [481, 105], [481, 90]]).T

    x = color_patches[0].astype(np.uint64).T
    y = color_patches[1].astype(np.uint64).T

    ones_array = np.ones((24, 1))
    coordinates = img_in[y, x, :]
    coordinates = np.hstack((coordinates, ones_array))

    r, g, b = read_colorchecker_gm()

    r = np.reshape(r, (24, 1))
    g = np.reshape(g, (24, 1))
    b = np.reshape(b, (24, 1))

    references = np.hstack((r, g, b, ones_array))
    x = np.linalg.lstsq(coordinates, references, rcond=None)[0]

    hdr_ones_array = np.ones((img_in.shape[0], img_in.shape[1]))
    hdr_homogeneous = np.dstack((img_in, hdr_ones_array))

    hdr_factor = hdr_homogeneous @ x
    hdr_color = hdr_factor[:, :, :3] / hdr_factor[:, :, 3:4]
    hdr_color = np.where(hdr_color < 0, 0, hdr_color)

    white_patch = color_patches[:, 18:24]
    white = hdr_color[white_patch].T

    red = hdr_color[:, :, 0] / np.average(white[0])
    green = hdr_color[:, :, 1] / np.average(white[1])
    blue = hdr_color[:, :, 2] / np.average(white[2])

    hdr_white_balance = np.dstack((red, green, blue))
    hdr_white_balance /= np.max(hdr_white_balance)
    hdr_white_balance = gamma_correction(hdr_white_balance)

    return hdr_white_balance, hdr_color


def gray_world(img_in):

    r_channel = img_in[:, :, 0]
    g_channel = img_in[:, :, 1]
    b_channel = img_in[:, :, 2]

    r_average = np.mean(r_channel)
    g_average = np.mean(g_channel)
    b_average = np.mean(b_channel)

    r_normalized = (r_channel * g_average) / r_average
    g_normalized = g_channel
    b_normalized = (b_channel * g_average) / b_average

    return np.dstack((r_normalized, g_normalized, b_normalized))


def white_world(img_in):

    r_channel = img_in[:, :, 0]
    g_channel = img_in[:, :, 1]
    b_channel = img_in[:, :, 2]

    r_average = np.mean(r_channel)
    g_average = np.mean(g_channel)
    b_average = np.mean(b_channel)

    max_average = max(r_average, g_average, b_average)
    r_scale = max_average / r_average
    g_scale = max_average / g_average
    b_scale = max_average / b_average

    r_balanced = r_channel * r_scale
    g_balanced = g_channel * g_scale
    b_balanced = b_channel * b_scale

    return np.dstack((r_balanced, g_balanced, b_balanced))


def XYZ2xyY(XYZ):
    X = XYZ[:, :, 0]
    Y = XYZ[:, :, 1]
    Z = XYZ[:, :, 2]

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    xyY = np.dstack((x, y, Y))
    return xyY


def xyY2XYZ(xyY):
    x = xyY[:, :, 0]
    y = xyY[:, :, 1]
    Y = xyY[:, :, 2]

    sum_XYZ = Y / y
    X = x * sum_XYZ
    Z = sum_XYZ - X - Y
    XYZ = np.dstack((X, Y, Z))
    return XYZ
