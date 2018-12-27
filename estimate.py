'''
Author - Pranav
'''

from lib.models.pose_resnet import get_pose_net
from lib.core.config import config
from lib.core.config import update_config
from lib.utils.transforms import transform_preds
from lib.core.inference import get_max_preds

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse

cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED

config.TEST.FLIP_TEST = True
update_config('pretrained/384x288_d256x3_adam_lr1e-3.yaml')
model = get_pose_net(config, is_train=False)
model.load_state_dict(torch.load('pretrained/pose_resnet_50_384x288.pth.tar'))

gpus = [int(i) for i in config.GPUS.split(',')]
model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
toTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean, std)])

def get_keypoints(input_image):
    '''
    Calculates keypoints based on resnet
    Input: Image
    Output: List of 19 Keypoints locations and probabilities
    '''
    H = 96
    W = 72
    img = cv2.imread(input_image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    height, width, channels = img.shape
    img = cv2.resize(img, (288, 384))
    x = toTensor(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        res = model(x)
        preds, maxvals = get_max_preds(res.clone().cpu().numpy())

    probs = []
    for prob in maxvals[0]:
        probs.append(prob[0])
    points = []
    for arr in preds[0]:
        points.append((arr[0], arr[1]))
    d1_x = (points[5][0] + points[6][0]) / 2
    d1_y = (points[5][1] + points[6][1]) / 2
    d2_x = (points[11][0] + points[12][0]) / 2
    d2_y = (points[11][1] + points[12][1]) / 2
    prob_17 = (probs[5] + probs[6]) / 2
    prob_18 = (probs[11] + probs[12]) / 2
    points.append((d1_x, d1_y))
    points.append((d2_x, d2_y))
    probs.append(prob_17)
    probs.append(prob_18)
    resize = []
    for coord in points:
        x = (coord[0] / W) * width
        y = (coord[1] / H) * height
        resize.append((x, y))
    return resize, probs

def draw(image, points, probs, res, threshold):
    '''
    Input: Image, Keypoints, Probabilities, Resolution, Threshold
    Draws keypoints on image if probability is greater than threshold
    '''
    rounded = []

    for point in points:
        x = int(round(point[0]))
        y = int(round(point[1]))
        rounded.append((x, y))

    def draw_line(index_1, index_2):
        if (rounded[index_1] > (0, 0)) and (rounded[index_2] > (0, 0)):
            if (probs[index_1] > threshold) and (probs[index_2] > threshold):
                cv2.line(image, rounded[index_1], rounded[index_2], (255, 255, 0), res)

    draw_line(0, 1)
    draw_line(0, 2)
    draw_line(1, 3)
    draw_line(2, 4)
    draw_line(0, 17)
    draw_line(17, 5)
    draw_line(17, 6)
    draw_line(6, 8)
    draw_line(8, 10)
    draw_line(5, 7)
    draw_line(7, 9)
    draw_line(17, 18)
    draw_line(18, 12)
    draw_line(18, 11)
    draw_line(12, 11)
    draw_line(12, 14)
    draw_line(14, 16)
    draw_line(11, 13)
    draw_line(13, 15)

    for i in range(19):
        if (rounded[i] > (0, 0)) and (probs[i] > threshold):
            cv2.circle(image, rounded[i], res, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)


parser = argparse.ArgumentParser()
parser.add_argument("image", help = "the image that you want to input")
parser.add_argument("--output", help = "the output filename", default = "output.jpg")
parser.add_argument("--threshold", help = "probability of the keypoint that should appear greater than this threshold", type = int, default = 0.1)
parser.add_argument("--thickness", help = "thickness of the line", type = int, default = 8)
args = parser.parse_args()
filename = args.image
keypoints, probs = get_keypoints(filename)
image = cv2.imread(filename)
draw(image, keypoints, probs, args.thickness, args.threshold)
cv2.imwrite(args.output, image)
