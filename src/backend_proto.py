import cv2 as cv
import numpy as np
from sklearn.preprocessing import normalize
import scipy
from matplotlib.pyplot import imread
import pickle as pickle
import random
import os
import matplotlib.pyplot as plt
from math import sqrt


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
    folder = [os.path.join(images_path, p)
              for p in sorted(os.listdir(images_path))]

    i = 0
    fileCount = 0
    for file in folder:
        result[fileCount] = extract_features(file)
        fileCount += 1
        
    return result


# MEAN
def mean(extraction_result):
    m = len(extraction_result)
    mean = [0 for i in range(2048)]
    for i in range(m):
        mean = np.add(mean, extraction_result[i])
    mean = np.divide(mean, m)

    return mean


# SELISIH (PHI)
def A(extraction_result, mean):
    m = len(extraction_result)
    A = [[0 for i in range(2048)] for j in range(m)]
    for i in range(m):
        A[i] = extraction_result[i] - mean

    return A


# MATRIX KOVARIAN
def kovarian(A):
    kovarian = np.matmul(A, np.transpose(A))
    n = len(kovarian)

    return kovarian


# EIGENFACE ALGORITHM
def getEigenFaces(eigenSpace, A, k):
    best = eigenSpace[0:k]
    bestOriEigenFace = np.matmul(best, A)

    return bestOriEigenFace


# WEIGHT SET CALCULATOR
def getWeightSet(A, eigenFaces, M):
    weightSet = [[0] for i in range(M)]
    for i in range(M):
        weightSet[i] = np.matmul(A[i], np.transpose(eigenFaces))

    return weightSet

# THRESHOLD
def getThreshold(eigenfaceWeight, M):
    for i in range(M):
        start = i+1
        sum = 0
        for k in range(len(eigenfaceWeight[i])):
            sum += eigenfaceWeight[i][k]**2
        magnitude1 = sqrt(sum)
        for j in range(start, M):
            sum = 0
            for k in range(len(eigenfaceWeight[j])):
                sum += eigenfaceWeight[j][k]**2
            magnitude2 = sqrt(sum)
            if (i == 0 and j == 1):
                leng = (eigenfaceWeight[j]/magnitude2) - \
                    (eigenfaceWeight[i]/magnitude1)
                max = np.dot(np.transpose(leng), leng)
            else:
                curr = (eigenfaceWeight[j]/magnitude2) - \
                    (eigenfaceWeight[i]/magnitude1)
                distance = np.dot(np.transpose(curr), curr)
                if (max < distance):
                    max = distance

    t = 0.5*max
    print("Ini threshold")
    print(t)
    return t

# MATCHER
def matcher(input, datasetName, mean, weightSet, M, threshold, eigenfaces):
    folder = [os.path.join(datasetName, p)
              for p in sorted(os.listdir(datasetName))]

    selisih = A([extract_features(input)], mean)
    weight = np.matmul(selisih[0], np.transpose(eigenfaces))
    sum = 0

    for k in range(len(weight)):
        sum += weight[k]**2
    magnitude1 = sqrt(sum)

    for i in range(M):
        sum = 0
        for k in range(len(weightSet[i])):
            sum += weightSet[i][k]**2
        magnitude2 = sqrt(sum)
        if (i == 0):
            sum = 0
            leng = (weightSet[i]/magnitude2) - (weight/magnitude1)
            min = np.dot(np.transpose(leng), leng)
            index = i
            distanceWeight = leng
        else:
            leng = (weightSet[i]/magnitude2) - (weight/magnitude1)
            distance = np.dot(np.transpose(leng), leng)
            if (distance < min):
                min = distance
                index = i
                distanceWeight = leng

    if (min > threshold):
        match = False
    else:
        match = True
        matchedPath = folder[index]

    return match, matchedPath, min, distanceWeight, weight

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


def qr(matrix):
    n = len(matrix)
    resultQ = [[0 for i in range(n)] for j in range(n)]
    resultQ = np.reshape(resultQ, (n, n))
    resultQ = resultQ.astype('float')
    arrTemp = [0 for i in range(n)]
    for i in range(len(matrix)):
        arrTemp = np.array(matrix[:, i])
        arrTemp = np.reshape(arrTemp, n)
        for j in range(0, i):
            u = np.array(resultQ[:, j])
            u = np.reshape(u, n)
            arrTemp = np.subtract(arrTemp, proj(u, arrTemp))
        for k in range(n):
            resultQ[k][i] = arrTemp[k]
    for x in range(n):
        arrTemp = np.array(resultQ[:, x])
        arrTemp = np.reshape(arrTemp, n)
        arrTemp = np.divide(arrTemp, norm(arrTemp))
        for y in range(n):
            resultQ[y][x] = arrTemp[y]
    resultR = [[0 for i in range(n)] for j in range(n)]
    resultR = np.reshape(resultR, (n, n))
    resultR = resultR.astype('float')
    transpose = np.transpose(resultQ)
    resultR = np.matmul(transpose, matrix)

    return resultQ, resultR


def qr2(A):
    A = np.array(A, dtype=type)
    n = len(A)
    Q = np.array(A, dtype=type)
    R = np.zeros((n, n), dtype=type)
    for i in range(n):
        for j in range(i):
            R[j, i] = np.transpose(Q[:, j]).dot(Q[:, i])
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]
        norm = 0
        for k in range(n):
            norm += (Q[k, i] ** 2)
        norm = norm ** (1/2)
        R[i, i] = norm
        Q[:, i] = Q[:, i] / R[i, i]
    return Q, R


def find_eig(matrix):
    n = len(matrix)
    Q = np.identity(n)
    Q, _ = qr2(Q)
    for i in range(100):
        Z = matrix.dot(Q)
        Q, R = qr2(Z)
    return np.diag(R), Q


def sorted_eig(matrix):
    eigVal, eigVec = find_eig(matrix)
    eigVec = np.transpose(eigVec)

    # Sorting from largest to smallest eigVal
    sortedEigVal = np.sort(eigVal)[::-1]
    sortedIdx = np.argsort(eigVal)[::-1]

    sortedEigVec = [eigVec[i] for i in sortedIdx]

    return sortedEigVal, sortedEigVec

# TEST
# x = np.array([[1, -2], [1, 4]])
# print(getEigenDiagonal(x))
# print(getEigenDiagonal(x))
# x = np.array([[0.5,0.25,0.25],[0.25,0.5,0.25],[0.25,0.25,0.5]])
# x = np.array([[4,0,1],[-2,1,0],[-2,0,1]])
# print(getEigenDiagonal(x))


# TEST CASES FOR getEigenSpace
# x = np.array([[3,0],[8,-1]])
# x = np.array([[1,3],[3,1]])
# x = np.array([[3,-2,0],[-2,3,0],[0,0,5]])
# x = np.array([[1,-2],[1,4]])
# x = np.array([[0.5,0.25,0.25],[0.25,0.5,0.25],[0.25,0.25,0.5]])
# x = np.array([[4,0,1],[-2,1,0],[-2,0,1]])
# e,v = find_eig(x)
# print(e)
# print(v)

# x = np.array([[1, -2, 4, 5], [-2, 0, 1, 7], [4, 1, -3, 4], [5, 7, 4, 1]])
# libe, libv = np.linalg.qr(x)
# eiglib, vlib = eig(x)
# print('lib: ')
# print(libv)
# print('eigenval1: ')
# print(eiglib)
# print(vlib)


# eigs, eigvs = qr2(x)
# eigsyey, v = find_eig(x)
# print('schurz: ')
# print(eigvs)
# print('eigenval2: ')
# print(eigsyey)
# print(v)
