# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:14:41 2019

@author: dcyrils
"""


data = pd.read_csv("E:/SNA9/train_data.csv", index_col=0)
votes2 = pd.read_csv("E:/SNA9/train2_data.csv", index_col=0)
answers = pd.read_csv("E:/SNA9/train_answers.csv", index_col=0)
votest = pd.read_csv("E:/SNA9/test1_data.csv", index_col=0)

data[(data.Xmin < data.Xmax) & (data.Ymin < data.Ymax)].shape
Out[221]: (5320, 5) !!!

data[(data.Xmin < data.Xmax) & (data.Ymin < data.Ymax)].to_csv("E:/SNA9/train_data.csv")

