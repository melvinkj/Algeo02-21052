import cv2 as cv
import numpy as np
import scipy
from matplotlib.pyplot import imread
import pickle as pickle
from scipy import spatial
import random
import os
import math
import matplotlib.pyplot as plt
import csv

# Feature extractor


def extract_features(image_path, vector_size=32):
    image = imread(image_path)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv.error as e:
        print('Error: ', e)
        return None
    return dsc


result = {}


def batch_extractor(images_path):
    result = {}
    folders = [os.path.join(images_path, p)
               for p in sorted(os.listdir(images_path))]

    i = 0
    for f in folders:
        files = [os.path.join(f, i) for i in sorted(os.listdir(f))]
        for n in files:
            name = n.split('\\')[-1]
            print('Extracting features from image %s' % name)
            result[name] = extract_features(n)
    return result


# temporary extractor buat ngetes
# result = batch_extractor("..\\test\\training_set")

# QR DECOMPOSITION
def norm(x):
    result = 0
    for i in x:
        result += (i**2)
    result = result ** (1/2)
    return result


def proj(u, v):
    result = 0
    norm_u2 = norm(u) ** 2
    for i in range(len(u)):
        result += (u[i] * v[i])
    result /= norm_u2
    result = np.multiply(result, u)
    return result


def qrGetQ(matrix):
    n = len(matrix)
    result = [[0 for i in range(n)] for j in range(n)]
    result = np.reshape(result, (n, n))
    result = result.astype('float')
    arrTemp = [0 for i in range(n)]
    for i in range(len(matrix)):
        arrTemp = np.array(matrix[:, i])
        arrTemp = np.reshape(arrTemp, n)
        for j in range(0, i):
            u = np.array(result[:, j])
            u = np.reshape(u, n)
            arrTemp = np.subtract(arrTemp, proj(u, arrTemp))
        for k in range(n):
            result[k][i] = arrTemp[k]
    for x in range(n):
        arrTemp = np.array(result[:, x])
        arrTemp = np.reshape(arrTemp, n)
        arrTemp = np.divide(arrTemp, norm(arrTemp))
        for y in range(n):
            result[y][x] = arrTemp[y]
    return result


def qrGetR(q, matrix):
    n = len(matrix)
    result = [[0 for i in range(n)] for j in range(n)]
    result = np.reshape(result, (n, n))
    transpose = np.transpose(q)
    result = np.matmul(transpose, matrix)
    return result


def cekTriangle(matrix):
    triangle = 1
    n = len(matrix)
    for i in range(1, n):
        for j in range(0, i):
            if (matrix[i][j] > 0.0001 or matrix[i][j] < -0.0001):
                triangle = 0
    return triangle


def getEigenDiagonal(matrix):
    triangle = 0
    while (triangle == 0):
        q = qrGetQ(matrix)
        r = qrGetR(q, matrix)
        x = np.matmul(r, q)
        n = len(matrix)
        triangle = cekTriangle(x)
        if (triangle == 0):
            matrix = x
    eigenVal = [0 for i in range(n)]
    eigenVal = np.array(eigenVal)
    eigenVal = eigenVal.astype('float')
    for i in range(n):
        eigenVal[i] = x[i][i]
    return eigenVal
