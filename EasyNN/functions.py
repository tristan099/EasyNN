"""
This script contains a variety of useful functions for pre-processing and regression tasks.
"""

# lib
import numpy as np


def GCEC(predicted, true):
    """
    Generalized Cross Entropy Cost for the case of multinomial
    classification.
    :param predicted: predicted values from model
    :param true: true values
    :return: errors
    """
    errs = -np.sum(true*np.log(predicted))/len(predicted)
    return(errs)

def BCEC(predicted, true):
    """
    Binary Cross entropy cost function for the case of two class classification
    :param predicted: predicted values from model
    :param true: true values
    :return: errors
    """
    errs = -np.sum(true*np.log(predicted)+(1-true) *
                   (np.log(1-predicted)))/len(predicted)
    return(errs)

def LCF(predicted, true):
    """
    Least Squares Cost function
    :param predicted: predicted values
    :param true: true values
    :return: errors
    """
    errs = ((predicted-true).T@(predicted-true))/len(predicted)
    return(errs)

def MCF(predicted, true):
    """
    Multinomial Cost function using Frobenius Norm
    :param predicted: predicted values
    :param true: true values
    :return: errors
    """
    errs = np.trace((predicted - true).T@(predicted - true))
    return(errs)

def SSE(predicted, true):
    """
    Sum of Squared Errors
    :param predicted: predicted values
    :param true: true values
    :return: errors
    """
    return(np.sum(np.multiply(true-predicted, true-predicted))/len(true[:, 0]))

def onehotencode(matrix, columns):
    """
    One hot encoding function for processing
    categorical data into numeric columns of 1's and 0's
    :param matrix: matrix containing data you want to one-hot encode
    :param columns: columns you want to one hot encode
    :return: original matrix with newly encoded columns
    """
    empty = np.matrix([])
    columnshape = matrix.shape[1]
    for i in range(columnshape):
        if i in columns:
            rows = len(matrix)
            z = np.unique(matrix[:, i])
            columnsunique = (len(z))
            x = np.zeros((rows, columnsunique))
            for v in range(0, len(matrix)):
                for j in range(0, len(z)):
                    if matrix[v, i] == z[j]:
                        x[v, j] = 1
                    else:
                        x[v, j] = 0
            if empty.size == 0:
                empty = x
            else:
                empty = np.hstack((empty, x))
        else:
            add = matrix[:, i]
            add = add.reshape((matrix.shape[0], 1))
            if empty.size == 0:
                empty = add
            else:
                empty = np.hstack((empty, add))
    return(empty)

def min_max_normalize(X):
    """
    Function to perform min max normalization on a matrix
    :param X: matrix to normalize
    :return: Normalized Matrix
    """
    return ((X-X.min(0))/(X.max(0)-X.min(0)))

def ohepseudo(matrix, columns):
    empty=np.matrix([])
    columnshape=matrix.shape[1]
    for i in range(columnshape):
        if i in columns:
            rows=len(matrix)
            z=np.unique(matrix[:, i])
            columnsunique=(len(z))
            for v in range(0, len(matrix)):
                for j in range(0, len(z)):
                    if matrix[v, i] == z[j]:
                        matrix[v, i]=list(z).index(j)
    return(matrix)
