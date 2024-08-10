import numpy as np
import matplotlib.pyplot as plt
from cp_hw2 import lRGB2XYZ
from scipy.signal import correlate2d
from scipy.interpolate import interp2d
from matplotlib.patches import Rectangle
import cv2
import warnings

# interp2d is depreciated so ignoring the warning
warnings.filterwarnings("ignore", category=DeprecationWarning)


def loadVid(video_path):
    cap = cv2.VideoCapture(video_path)

    i = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if i == 1:
                frames = frame[np.newaxis, ...]
            else:
                frame = frame[np.newaxis, ...]
                frames = np.vstack([frames, frame])
                frames = np.squeeze(frames)
        else:
            break

    cap.release()
    return frames


def crossCorrelation(frame, template, xt, yt):
    template = lRGB2XYZ(template)[:, :, 1]
    frame = lRGB2XYZ(frame)[:, :, 1]

    g = template[xt:(xt + 50), yt:(yt + 50)]
    g_mean = np.mean(g)

    I = frame[(xt - 100):(xt + 150), (yt - 100): (yt + 150)]
    I_mean = np.mean(I)

    h = correlate2d(I - I_mean, g - g_mean, mode='same') / np.sqrt(np.sum((g - g_mean) ** 2) * np.sum((I - I_mean) ** 2))

    return np.unravel_index(np.argmax(h), h.shape)


def interp(im, x, y):
    w = np.arange(im.shape[1])
    h = np.arange(im.shape[0])
    i = interp2d(w, h, im)
    int = i(w + x, h + y)
    return int


def refocusUnstructured(template, xt, yt, vid):
    img = np.zeros_like(vid[0])
    count = 0

    for frame in vid:
        count += 1
        sx, sy = crossCorrelation(frame, template, xt, yt)

        sx -= 25
        sy -= 25

        imR = frame[:, :, 0]
        imG = frame[:, :, 1]
        imB = frame[:, :, 2]

        iR = interp(imR, sy, sx)
        iG = interp(imG, sy, sx)
        iB = interp(imB, sy, sx)

        shifted = np.dstack((iR, iG, iB)).astype(np.uint16)
        img = (img + shifted)

    img = img / (count * 255)
    return img


video = loadVid('./data/mug2.MOV')
middleFrame = len(video) // 2
template = video[middleFrame]

x1 = 575
y1 = 1100

x2 = 520
y2 = 1200

x3 = 630
y3 = 1300

x4 = 300
y4 = 820

# plt.imshow(template)
# rect = Rectangle((x1, y1), 50, 50, linewidth=2, edgecolor='r', facecolor='none')
# plt.gca().add_patch(rect)
# # plt.savefig('./results/focus_template3.png')
# plt.show()

# plt.imshow(template[y:y + height, x:x + width])
# plt.savefig('./results/focus_template_cropped3.png')

refocus1 = refocusUnstructured(template, y1, x1, video)
plt.imsave('./results/template_refocused1.png', refocus1)

# refocus2 = refocusUnstructured(template, y2, x2, video)
# plt.imsave('./results/template_refocused2.png', refocus2)

# refocus3 = refocusUnstructured(template, y3, x3, video)
# plt.imsave('./results/template_refocused3.png', refocus3)

# refocus4 = refocusUnstructured(template, y4, x4, video)
# plt.imsave('./results/template_refocused4.png', refocus4)
