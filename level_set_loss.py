from torch import nn
import torch
import cv2
import  numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage import io
from PIL import Image
from pylab import *
import math

class LS_loss(nn.Module):
    def __init__(self, epison=1/20.):
        super().__init__()
        self.epison = epison

    def forward(self, input, img_mean):
        Hea = 1 / 2 * (1 + torch.tanh(input / self.epison))
        c1, c2 = self.calculate_constant(input, img_mean)
        c1 = np.repeat(c1, img_mean.shape[2] * img_mean.shape[3], -1).reshape(img_mean.shape)
        c2 = np.repeat(c2, img_mean.shape[2] * img_mean.shape[3], -1).reshape(img_mean.shape)
        c1, c2 = torch.tensor(c1, dtype=input.dtype).to(input.device), torch.tensor(c2, dtype=input.dtype).to(input.device)
        sqr1 = (img_mean - c1)**2
        sqr2 = (img_mean - c2)**2

        In = Hea * sqr1
        Out = (1 - Hea) * sqr2

        loss = In + Out
        return torch.mean(loss)

    def divide(self, img, reduction_ratio):
        img = img.detach().to('cpu').numpy()
        img = np.mean(img, axis=1, keepdims=True)
        row_parts = img.shape[2] // reduction_ratio
        col_parts = img.shape[3] // reduction_ratio
        mean = np.zeros((img.shape[0], img.shape[1], row_parts, col_parts))
        for row in range(row_parts):
            for col in range(col_parts):
                part = img[:, :, row * reduction_ratio:(row+1) * reduction_ratio, col * reduction_ratio:(col+1) * reduction_ratio]
                part = np.reshape(part, (part.shape[0], part.shape[1], -1))
                mean[:, :, row, col] = np.mean(part, axis=-1)
        img_mean = mean
        return img_mean


    def calculate_constant(self, input, img):
        input  = input.detach().to('cpu').numpy()
        Hea = 1/2 * (1 + np.tanh(input/self.epison))
        img = img.detach().to('cpu').numpy()
        Hea = Hea.reshape((Hea.shape[0], Hea.shape[1], -1))
        img = img.reshape((img.shape[0], img.shape[1], -1))
        s1 = Hea * img
        s2 = (1-Hea) * img
        s3 = 1 - Hea

        e = 0.0001
        c1 = s1.sum(axis=-1) / (Hea.sum(axis=-1) + e)
        c2 = s2.sum(axis=-1) / (s3.sum(axis=-1) + e)
        return c1, c2



def divide3(img):
    mean_4 = divide_mean(img, 4)
    mean_8 = divide_mean(img, 8)
    mean_16 = divide_mean(img, 16)
    return (mean_4, mean_8, mean_16)

def divide_mean(img_gpu, reduction_ratio):
    img = img_gpu.detach().to('cpu').numpy()
    img = np.mean(img, axis=1, keepdims=True)
    row_parts = img.shape[2] // reduction_ratio
    col_parts = img.shape[3] // reduction_ratio
    mean = np.zeros((img.shape[0], img.shape[1], row_parts, col_parts))
    for row in range(row_parts):
        for col in range(col_parts):
            part = img[:, :, row * reduction_ratio:(row+1) * reduction_ratio, col * reduction_ratio:(col+1) * reduction_ratio]
            part = np.reshape(part, (part.shape[0], part.shape[1], -1))
            mean[:, :, row, col] = np.mean(part, axis=-1)
    img_mean = mean
    return torch.tensor(img_mean, dtype=img_gpu.dtype, device=img_gpu.device)



