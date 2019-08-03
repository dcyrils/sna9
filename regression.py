# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:37:24 2019

@author: dcyrils
"""

%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
pd.set_option('display.max_columns', 999)

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def intersection_over_union(boxes):
    assert(len(boxes) == 8)
    boxA = boxes[:4].values
    boxB = boxes[4:].values
    
    boxAArea = area(boxA)
    boxBArea = area(boxB)
    
    if (boxAArea == 0 or boxBArea == 0):
        return 0
        
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

votes = pd.read_csv("E:/SNA9/train1_data.csv", index_col=0)
votes2 = pd.read_csv("E:/SNA9/train2_data.csv", index_col=0)
answers = pd.read_csv("E:/SNA9/train_answers.csv", index_col=0)
votest = pd.read_csv("E:/SNA9/test1_data.csv", index_col=0)

""" Задам квадраты погрешностей между тру и рил """
def SquaredError(A, B, C):
    votes2[A] = (votes2[B] - votes2[C])**2
SquaredError('se_Xmin', 'Xmin', 'Xmin_true')
SquaredError('se_Ymin', 'Ymin', 'Ymin_true')
SquaredError('se_Xmax', 'Xmax', 'Xmax_true')
SquaredError('se_Ymax', 'Ymax', 'Ymax_true')







