# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score

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

votes = pd.read_csv("E:/SNA9/train_data.csv")
answers = pd.read_csv("E:/SNA9/train_answers.csv")
votest = pd.read_csv("E:/SNA9/test_data.csv")

def Trues(A):
    votes[A] = 0
    for i in range(len(votes)):
        for k in range(len(answers)):
            if votes.itemId[i] == answers.itemId[k]:
                votes[A][i] = answers[A][k]

Trues('Ymin_true')
Trues('Xmax_true')
Trues('Ymax_true')

votes.to_csv("E:/SNA9/train2_data.csv")

acc = []
for i in range(len(votes)):
    y_true = [votes.Xmin_true[i], votes.Ymin_true[i], votes.Xmax_true[i], votes.Ymax_true[i]]
    y_test = [votes.Xmin[i], votes.Ymin[i], votes.Xmax[i], votes.Ymax[i]]
    acc.append(explained_variance_score(y_true, y_test))
    
Acc = pd.Series(acc)    
votes['acc'] = Acc

votes.shape
votes_good = votes[votes.acc > 0.96]

""" true moment """
quorum = votes_good.groupby("itemId")[['Xmin','Ymin', 'Xmax', 'Ymax']].median().reset_index()
data = quorum.merge(answers, on=["itemId"])
data["iou"] = data[['Xmin','Ymin', 'Xmax', 'Ymax', 'Xmin_true',\
      'Ymin_true', 'Xmax_true','Ymax_true']].apply(intersection_over_union, axis=1)
data["iou"].mean() #0.5940656242240981 NOT BAD!!!

good_fellows = set(votes_good.userId.unique())
votest.shape #3615
votest_good = votest[votest.userId.isin(good_fellows)] #.shape # 3015

quo_test = votest_good.groupby("itemId")[['Xmin','Ymin', 'Xmax', 'Ymax']].median().reset_index()
quo_test.shape
Out[84]: (629, 5) - 630!

median1 = pd.read_csv("E:/SNA9/median1.csv", header=None)
median1[[0]].sum() - quo_test.itemId.sum()  # 13369
set(median1[0]) - set(quo_test.itemId)
diff = median1[median1[0].isin(set(median1[0]) - set(quo_test.itemId))]
diff = diff.rename(columns={0: "itemId", 1: "Xmin", 2: "Ymin", 3: "Xmax", 4: "Ymax"})

quo_test = pd.concat([quo_test, diff])


quo_test.set_index('itemId').sort_index().to_csv("E:/SNA9/goodfellas99.csv", header=None)

