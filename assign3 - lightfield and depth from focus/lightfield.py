import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy import interpolate
from cp_hw2 import lRGB2XYZ
import warnings
import cv2

# interp2d is depreciated so ignoring the warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Given as a hint
lensletSize = 16
maxUV = (lensletSize - 1) / 2
u = np.arange(lensletSize) - maxUV
v = np.arange(lensletSize) - maxUV

# Part 1: Initials
im = io.imread('data/chessboard_lightfield.png')

s = im.shape[1] // lensletSize
t = im.shape[0] // lensletSize

L = np.transpose((im.reshape(lensletSize, t, lensletSize, s, 3, order='F')), (2, 0, 1, 3, 4)).astype(np.uint8)

# Part 2: Sub-aperture views
L_mosaic = np.transpose(L, (1, 0, 2, 3, 4))
mosaic = np.transpose(L_mosaic, (0, 2, 1, 3, 4)).reshape(lensletSize * L.shape[2], lensletSize * L.shape[3], 3)

plt.imshow(mosaic)
plt.savefig('results/mosaic.png')

# Part 3: Refocusing and focal-stack simulation
def refocus(img, depth):
    refocus = np.zeros((img.shape[2], img.shape[3], img.shape[4]))

    for i in range(len(u)):
        for j in range(len(v)):
            print("Shape of img:", img.shape)
            red_channel = img[i, j, :, :, 0]
            green_channel = img[i, j, :, :, 1]
            blue_channel = img[i, j, :, :, 2]
            print("Shape of red_channel:", red_channel.shape)
            print("Shape of green_channel:", green_channel.shape)
            print("Shape of blue_channel:", blue_channel.shape)

            s_red = np.arange(red_channel.shape[1])
            t_red = np.arange(red_channel.shape[0])
            interp_red = interpolate.interp2d(s_red, t_red, red_channel)
            interp_red_image = interp_red(s_red + (depth * u[i]), t_red - (depth * v[j]))

            s_green = np.arange(green_channel.shape[1])
            t_green = np.arange(green_channel.shape[0])
            interp_green = interpolate.interp2d(s_green, t_green, green_channel)
            interp_green_image = interp_green(s_green + (depth * u[i]), t_green - (depth * v[j]))

            s_blue = np.arange(blue_channel.shape[1])
            t_blue = np.arange(blue_channel.shape[0])
            interp_blue = interpolate.interp2d(s_blue, t_blue, blue_channel)
            interp_blue_image = interp_blue(s_blue + (depth * u[i]), t_blue - (depth * v[j]))

            refocus += np.dstack((interp_red_image,
                                  interp_green_image,
                                  interp_blue_image))

    return (refocus / (lensletSize ** 2)) / 255


depths = [-2.0, -1.5, -1.0, 0.0, 0.5]
for i in range(len(depths)):
    r = refocus(L, depths[i])
    fig = plt.figure()
    plt.imshow(r)
    plt.title(f'Depth={depths[i]}')
    plt.savefig(f'results/depth{i + 1}.png')

# Part 4: All-in-focus image and depth from focus
def gamma(img):
    return np.where(img <= 0.0404482, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def allInFocus(img):
    ds = np.linspace(-2.0, 2.0, lensletSize)
    stack = np.zeros((img.shape[2], img.shape[3], img.shape[4], len(ds)))
    fs = stack

    k1 = 5
    k2 = 39

    s1 = 0
    s2 = 0

    for d in range(len(ds)):
        stack[:, :, :, d] = refocus(L, ds[d])

    xyz = np.zeros(fs.shape)
    for i in range(len(ds)):
        xyz[:, :, :, i] = lRGB2XYZ(gamma(fs[:, :, :, i]))

    luminance = xyz[:, :, 1, :]

    low_freq = cv2.GaussianBlur(luminance, [k1, k1], s1)
    high_freq = luminance - low_freq

    weight = cv2.GaussianBlur(pow(high_freq, 2), [k2, k2], s2)
    weight = np.stack((weight, weight, weight), axis=2)

    ds = np.asarray(ds + abs(np.min(ds)))
    ds = ds / np.max(ds)

    focusedNumerator = np.sum(weight * fs, axis=-1)
    focusedDenominator = np.sum(weight, axis=-1)

    depthNumerator = np.sum(weight * ds[None, None, None, :], axis=-1)
    depthDenominator = np.sum(weight, axis=-1)

    focused = np.nan_to_num(np.divide(focusedNumerator, focusedDenominator))
    depth = np.nan_to_num(np.divide(depthNumerator, depthDenominator))

    return focused, depth


# focused_image, depth_map = allInFocus(L)
#
# plt.imshow(focused_image)
# plt.title('All in Focus Image')
# plt.savefig('results/focused_image.png')
#
# plt.imshow(depth_map)
# plt.title('Depth Map')
# plt.savefig('results/depth_map.png')
