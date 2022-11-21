import cv2 as cv
import numpy as np
# import sympy as sp
# from sympy import Matrix
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
        # name = file.split('\\')[-1]
        # print('Extracting features from image %s' % name)
        result[fileCount] = extract_features(file)
        fileCount += 1
    return result


# temporary extractor buat ngetes
# result = batch_extractor("..\\test\\training_set\\")


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
    # for i in range(n):
    #     for j in range(n):
    #         kovarian[i][j] = round(kovarian[i][j], 3)
    return kovarian

# EIGENFACE ALGORITHM


def getEigenFaces(eigenSpace, A, k):
    best = eigenSpace[0:k]
    # print(np.array(best).shape)
    # print(np.array(A).shape)
    # print("Ini best")
    # print(np.array(best).shape)
    # print(best)
    # print("Ini A")
    # print(np.array(A).shape)
    # print(A)
    bestOriEigenFace = np.matmul(best, A)
    # print(bestOriEigenFace)
    # print("Ini best ori")
    # print(bestOriEigenFace)
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
                # leng = np.linalg.norm(eigenfaceWeight[j]) - np.linalg.norm(eigenfaceWeight[i])
                leng = (eigenfaceWeight[j]/magnitude2) - \
                    (eigenfaceWeight[i]/magnitude1)
                max = np.dot(np.transpose(leng), leng)
            else:
                curr = (eigenfaceWeight[j]/magnitude2) - \
                    (eigenfaceWeight[i]/magnitude1)
                # curr = eigenfaceWeight[j] - eigenfaceWeight[i]
                distance = np.dot(np.transpose(curr), curr)
                if (max < distance):
                    max = distance

    t = 0.5*max
    print("Ini threshold")
    print(t)
    return t

# MATCHER


# def matcher(input, datasetName, mean, weightSet, M, threshold, eigenfaces):
#     folder = [os.path.join(datasetName, p)
#               for p in sorted(os.listdir(datasetName))]
#     # perlu transpose input dan training set dulu
#     # print(np.array(mean).shape)
#     # print(mean)
#     selisih = A([extract_features(input)], mean)
#     # print(np.array(selisih).shape)
#     weight = [0 for i in range(M)]

#     weight = np.matmul(selisih[0], np.transpose(eigenfaces))
#     # print(weight)
#     # print(np.array(weight).shape)

#     for i in range(M):
#         # Normalisasi vektor
#         # norm_i = np.linalg.norm(weightSet[i])
#         # norm = np.linalg.norm(weight)
#         # vecNorm_i = np.multiply(weightSet[i], 1/norm_i)
#         # vecNorm = np.multiply(weight, 1/norm)

#         # Formula Euclidean Distance
#         subResult = np.subtract(weightSet[i], weight)
#         normResult = np.linalg.norm(subResult)
#         distance = (normResult**2)/2

#         # Rumus cos(teta) = 1 - ||x-x'||^2 / 2 = cos (teta)
#         # nilai distance adalah ||x-x'|| ^2 /2 , boleh lebih dari 1. Namun bila lebih dari satu maka dikurangi 1 (mendekati cos 0)
#         if (distance > 1):
#             distance -=1

#         print("index", i+1, ": ", distance)
#         if (i == 0):
#             min = distance
#             # min = np.linalg.norm(np.subtract(weight, weightSet[i]))
#             # print("distance ", i, ": ")
#             # print(min)
#             index = i
#             print("Ini distance ", str(i), " : ", str(min))

#         else:
#             leng = np.linalg.norm(weightSet[i]) - np.linalg.norm(weight)
#             distance = np.dot(np.transpose(leng), leng)
#             print("Ini distance ", str(i), " : ", str(distance))
#             # distance = np.linalg.norm(np.subtract(weight, weightSet[i]))
#             # print("distance ", i, ":")
#             # print(distance)
#             if (distance < min):
#                 min = distance
#                 index = i

#     if (min > threshold):
#         # Kasus tidak ada yang mirip
#         match = False
#     else:
#         match = True
#         matchedPath = folder[index]


#     return match, matchedPath, min
def matcher(input, datasetName, mean, weightSet, M, threshold, eigenfaces):
    folder = [os.path.join(datasetName, p)
              for p in sorted(os.listdir(datasetName))]
    # perlu transpose input dan training set dulu
    # print(np.array(mean).shape)
    # print(mean)
    selisih = A([extract_features(input)], mean)
    # print("Ini selisih:")
    # print(selisih)
    # print(np.array(selisih).shape)
    weight = [0 for i in range(M)]
    # print("Ini eigenfaces:")
    # print(eigenfaces)

    weight = np.matmul(selisih[0], np.transpose(eigenfaces))
    # print("Ini weight: ")
    # print(weight)
    # print(weight)
    # print(np.array(weight).shape)
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
            # leng = np.linalg.norm(weightSet[i]) - np.linalg.norm(weight)
            leng = (weightSet[i]/magnitude2) - (weight/magnitude1)
            min = np.dot(np.transpose(leng), leng)
            # min = np.linalg.norm(np.subtract(weight, weightSet[i]))
            # print("distance ", i, ": ")
            # print(min)
            index = i
            distanceWeight = leng
        else:
            # leng = np.linalg.norm(weightSet[i]) - np.linalg.norm(weight)
            leng = (weightSet[i]/magnitude2) - (weight/magnitude1)
            distance = np.dot(np.transpose(leng), leng)
            print(distance)
            # print("Ini min")
            # print(min)
            # print("ini distance")
            # print(distance)
            # distance = np.linalg.norm(np.subtract(weight, weightSet[i]))
            # print("distance ", i, ":")
            # print(distance)
            if (distance < min):
                min = distance
                index = i
                distanceWeight = leng
        # if (min == 0):
        #     break

    if (min > threshold):
        match = False
    else:
        match = True
        matchedPath = folder[index]

    print("Ini distance")
    print(min)
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


def conj_transpose(a):
    return np.conj(np.transpose(a))


def proj_ab(a, b):
    return a.dot(conj_transpose(b)) / np.linalg.norm(b) * b


def qr_gs(A, type=complex):
    A = np.array(A, dtype=type)
    (m, n) = np.shape(A)

    (m, n) = (m, n) if m > n else (m, m)

    Q = np.zeros((m, n), dtype=type)
    R = np.zeros((m, n), dtype=type)

    for k in range(n):
        pr_sum = np.zeros(m, dtype=type)

        for j in range(k):
            pr_sum += proj_ab(A[:, k], Q[:, j])

        Q[:, k] = A[:, k] - pr_sum
        Q[:, k] = Q[:, k] / np.linalg.norm(Q[:, k])

    if type == complex:
        R = conj_transpose(Q).dot(A)
    else:
        R = np.transpose(Q).dot(A)

    return -Q, -R
# def qrGetR(q, matrix):
#     n = len(matrix)
#     result = [[0 for i in range(n)] for j in range(n)]
#     result = np.reshape(result, (n, n))
#     transpose = np.transpose(q)
#     result = np.matmul(transpose, matrix)
#     return result


# def cekTriangle(matrix):
#     triangle = 1
#     n = len(matrix)
#     for i in range(1, n):
#         for j in range(0, i):
#             if (matrix[i][j] > 0.0001 or matrix[i][j] < -0.0001):
#                 triangle = 0
#     return triangle

# def find_eig(matrix):
#     n, m = matrix.shape
#     Qdot = np.eye(n)
#     Q = qrGetQ(matrix)
#     R = qrGetR(Q, matrix)
#     for i in range(100):
#         Z = R.dot(Q)
#         Qdot = Qdot.dot(Q)
#         Q = qrGetQ(Z)
#         R = qrGetR(Q, Z)
#     return np.diag(Z), Qdot


def find_eig(matrix):
    n, m = matrix.shape
    Q = np.random.rand(n, n)
    # Q, _ = np.linalg.qr(Q)
    Q, _ = qr(Q)
    # Q, R = np.linalg.qr(matrix)
    for i in range(100):
        Z = matrix.dot(Q)
        # Q, R = np.linalg.qr(Z)
        Q, R = qr(Z)
    return np.diag(Z), Q


def sorted_eig(matrix):
    eigVal, eigVec = find_eig(matrix)
    eigVec = np.transpose(eigVec)
    # for i in range (len(eigVec)):
    #     eigVec[i] = np.linalg.norm(eigVec[i])

    # Sorting from largest to smallest eigVal
    sortedEigVal = np.sort(eigVal)[::-1]
    sortedIdx = np.argsort(eigVal)[::-1]

    sortedEigVec = [eigVec[i] for i in sortedIdx]

    # sortedEigVec = np.transpose(sortedEigVec)

    return sortedEigVal, sortedEigVec

# def getEigenDiagonal(matrix):
#     triangle = 0
#     while (triangle == 0):
#         q = qrGetQ(matrix)
#         r = qrGetR(q, matrix)
#         x = np.matmul(r, q)
#         n = len(matrix)
#         triangle = cekTriangle(x)
#         if (triangle == 0):
#             matrix = x
#     eigenVal = [0 for i in range(n)]
#     eigenVal = np.array(eigenVal)
#     eigenVal = eigenVal.astype('float')
#     for i in range(n):
#         eigenVal[i] = x[i][i]
#         eigenVal[i] = round(eigenVal[i], 3)

#     eigenVal=sorted(eigenVal.tolist(), reverse=True)
#     return eigenVal


# def getEigenSpace(matrix):
#     # eigenVal = [3, -1]
#     # eigenVal = [4,-2]
#     # eigenVal = [5,5,1]
#     # eigenVal = [3, 2]
#     # eigenVal = [1,1,1]
#     # eigenVal = [3,2,1]

#     eigenVal = getEigenDiagonal(matrix)
#     n = len(matrix)
#     identity = np.identity(n)
#     repeat = 0              # a variable for containing the iteration of repeating eigen value
#     for i in range(n):
#         lamda = eigenVal[i]

#         # Generating lamda.I Matrix
#         lamdaI = lamda * identity
#         # Generating lamda.I - A Matrix
#         m = np.subtract(lamdaI, matrix)
#         m = Matrix(m)

#         # Getting Nullspace of m to Solve Parametric Equation
#         v = m.nullspace()
#         v = np.transpose(Matrix(v))

#         # Inserting Eigen Vector to Eigen Space
#         if (i == 0):
#             e = v
#             if (len(v[0]) > n):
#                 e = [v[0][:n]]
#                 repeat+=1
#         else:
#             if (len(v[0]) > n):
#                 v = [v[0][repeat*n:(repeat+1)*n]]
#                 # e = np.concatenate((e,v), axis = 1)
#                 # e = np.hstack(e,v)
#                 # e = np.c_[e, v]
#                 e = np.concatenate((e, v), axis=0)
#                 repeat+=1
#             else:
#                 # e = np.concatenate((e,v), axis = 1)
#                 # e = np.hstack(e,v)
#                 # e = np.c_[e, v]
#                 e = np.concatenate((e, v), axis=0)
#                 repeat = 0

#     return e


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
