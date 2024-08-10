import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from cp_hw2 import lRGB2XYZ
from skimage import io
from cv2 import GaussianBlur
from cp_hw4 import integrate_frankot, integrate_poisson, load_sources
import math

# Initials
path = './data/input_'
I_stack = []

for i in range(1, 8):

    file = path + str(i) + '.tif'
    image = io.imread(file).astype(np.float64)
    image /= 2 ** 16 - 1

    if image.shape[0] > 512 or image.shape[1] > 512:

        if image.shape[0] > image.shape[1]:
            d = math.ceil(image.shape[0] / 512)
        else:
            d = math.ceil(image.shape[1] / 512)

        image = image[::d, ::d]

    h = image.shape[0]
    w = image.shape[1]
    l1 = np.ravel(lRGB2XYZ(image)[:, :, 1])

    I_stack.append(l1)

I_stack = np.asarray(I_stack)
#######

# Getting Normals
u, s, v = np.linalg.svd(I_stack, full_matrices=False)

s = np.diag(s)
s = np.sqrt(s)

b = np.matmul(s, v)
b = b[:3]

l = np.matmul(s, u)
l = l[:3]

a = np.linalg.norm(b, axis=0)

n = b / a
#######

def visualize(a, n, h, w, cal):
    n = np.transpose(n)

    n = np.reshape(n, (h, w, 3))
    a = np.reshape(a, (h, w))

    n = (n - (np.min(n)))
    n = n / np.max(n)

    plt.figure()
    plt.imshow(n)
    plt.axis('off')
    if cal:
        plt.title('Calibrated Normal')
        plt.savefig('./results/calibrated_normal.png')
    else:
        plt.title('Uncalibrated Normal')
        plt.savefig('./results/uncalibrated_normal.png')
    # plt.show()

    plt.imshow(a, cmap='gray')
    plt.axis('off')
    if cal:
        plt.title('Calibrated Albedo')
        plt.savefig('./results/calibrated_albedo.png')
    else:
        plt.title('Uncalibrated Albedo')
        plt.savefig('./results/uncalibrated_albedo.png')
    # plt.show()


visualize(a, n, h, w, False)

def visualizeQ(b, h, w):
    q = np.array([[8, 9, 7], [1, 2, 3], [9, 2, 1]])
    q = np.transpose(q)
    q = np.linalg.inv(q)

    bq = np.matmul(q, b)

    a = np.linalg.norm(bq, axis=0)
    n = bq / (a + 0.000001)

    n = np.transpose(n)

    n = np.reshape(n, (h, w, 3))
    a = np.reshape(a, (h, w))

    n = (n - (np.min(n)))
    n = n / np.max(n)

    plt.figure()

    plt.imshow(n)
    plt.axis('off')
    plt.title('Uncalibrated Normal BQ')
    plt.savefig('./results/uncalibrated_normal_BQ.png')
    # plt.show()

    plt.imshow(a, cmap='gray')
    plt.axis('off')
    plt.title('Uncalibrated Albedo BQ')
    plt.savefig('./results/uncalibrated_albedo_BQ.png')
    # plt.show()


visualizeQ(b, h, w)

# Enforcing Integrability
k = 41
s = 10
g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

be = np.transpose(b)
be = np.reshape(be, (h, w, 3))

be0 = be[:, :, 0]
be1 = be[:, :, 1]
be2 = be[:, :, 2]

bg = np.zeros((h, w, 3))

for c in range(3):
    bg[:, :, c] = GaussianBlur(be[:, :, c], (k, k), s)

by0, bx0 = np.gradient(bg[:, :, 0], edge_order=2)
by1, bx1 = np.gradient(bg[:, :, 1], edge_order=2)
by2, bx2 = np.gradient(bg[:, :, 2], edge_order=2)

A1 = np.asarray(be0 * bx1 - be1 * bx0)
A2 = np.asarray(be0 * bx2 - be2 * bx0)
A3 = np.asarray(be1 * bx2 - be2 * bx1)
A4 = np.asarray(-be0 * by1 + be1 * by0)
A5 = np.asarray(-be0 * by2 + be2 * by0)
A6 = np.asarray(-be1 * by2 + be2 * by1)

A = np.hstack((A1.reshape(-1, 1), A2.reshape(-1, 1), A3.reshape(-1, 1), A4.reshape(-1, 1), A5.reshape(-1, 1),
               A6.reshape(-1, 1)))

u, s, v = np.linalg.svd(A, full_matrices=False)
x = v[-1, :]

delta = np.asarray([[-x[2], x[5], 1], [x[1], -x[4], 0], [-x[0], x[3], 0]])

bDelta = np.matmul(np.linalg.inv(np.transpose(g)), np.matmul(np.linalg.inv(delta), b))
aDelta = np.linalg.norm(bDelta, axis=0, keepdims=True)
nDelta = bDelta / (aDelta + 0.000001)
#######

def visualizeEnforced(a, n, h, w):
    n = np.transpose(n)

    n = np.reshape(n, (h, w, 3))
    a = np.reshape(a, (h, w))

    n = (n - (np.min(n)))
    n = n / np.max(n)

    plt.figure()
    plt.imshow(n)
    plt.axis('off')
    plt.title('Uncalibrated Normal After Enforcing Integrability')
    plt.savefig('./results/uncalibrated_normal_enforced.png')

    plt.imshow(a, cmap='gray')
    plt.axis('off')
    plt.title('Uncalibrated Albedo After Enforcing Integrability')
    plt.savefig('./results/uncalibrated_albedo_enforced.png')
    # plt.show()


visualizeEnforced(aDelta, nDelta, h, w)

def normalIntegration(n, h, w, e):
    n = np.transpose(n)
    n = np.reshape(n, (h, w, 3))

    dx = n[:, :, 0] / (n[:, :, 2] + e)
    dy = n[:, :, 1] / (n[:, :, 2] + e)

    sPoisson = integrate_poisson(dx, dy)
    sPoisson = sPoisson - np.min(sPoisson)
    sPoisson = (sPoisson / np.max(sPoisson))

    sFrankot = integrate_frankot(dx, dy)
    sFrankot = sFrankot - np.min(sFrankot)
    sFrankot = (sFrankot / np.max(sFrankot))

    return sPoisson, sFrankot


poisson, frankot = normalIntegration(nDelta, h, w, 0.0000001)

plt.imshow(-frankot, cmap='binary')
plt.axis('off')
plt.title('Uncalibrated Depth')
plt.savefig('./results/uncalibrated_depth.png')
# plt.show()

x, y = np.meshgrid(np.arange(h), np.arange(w))

# set 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# add a light and shade to the axis for visual effect
ls = LightSource()
color_shade = ls.shade(-frankot, plt.cm.gray)

# display a surface
ax.plot_surface(np.transpose(x), np.transpose(y), -frankot, facecolors=color_shade, rstride=4, cstride=4)

# turn off axis
plt.axis('off')
plt.title('Uncalibrated 3D')
plt.savefig('./results/uncalibrated_3D.png')
# plt.show()
#######

# Calibrated Photometric Stereo
g2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

sources = load_sources()
sourcesInv = np.linalg.pinv(sources)

bCal = np.matmul(sourcesInv, I_stack)
bCal = np.matmul(np.linalg.inv(np.transpose(g2)), bCal)

aCal = np.linalg.norm(bCal, axis=0)
nCal = bCal / (aCal + 0.00001)

visualize(aCal, nCal, h, w, True)

poissonCal, frankotCal = normalIntegration(nCal, h, w, 0.0000001)

plt.figure()
plt.imshow(frankotCal, cmap='binary')
plt.axis('off')
plt.title('Calibrated Depth')
plt.savefig('./results/calibrated_depth.png')
# plt.show()

x, y = np.meshgrid(np.arange(h), np.arange(w))

# set 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# add a light and shade to the axis for visual effect
ls = LightSource()
color_shade = ls.shade(frankotCal, plt.cm.gray)

# display a surface
ax.plot_surface(np.transpose(x), np.transpose(y), frankotCal, facecolors=color_shade, rstride=4, cstride=4)

# turn off axis
plt.axis('off')
plt.title('Calibrated 3D')
plt.savefig('./results/calibrated_3D.png')
# plt.show()
#######
