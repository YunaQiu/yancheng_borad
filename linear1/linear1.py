#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： 线性回归
模型参数： 无
特征： 基于周数和星期的修正日期（day）
      星期的one-hot特征
结果： A榜866352

'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib.dates
from datetime import *

from sklearn.preprocessing import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# 导入数据
def importDf(url):
    df = pd.read_csv(url, sep='\t')    
    return df

# 标记周数
def tickWeek(df, start):
    preVal = df[:-1]['day_of_week'].values
    preVal = np.insert(preVal, 0, preVal[0])
    df['week'] = (preVal > df.day_of_week)
    weekList = list()
    week = start
    for item in df['week']:
        week += (1 if item else 0)
        weekList.append(week)
    df['week'] = weekList    
    return df

# 计算统计量
def statCnt(cntArr):
    cntArr = np.array(cntArr)
    mean = cntArr.mean()
    square = np.square(cntArr).mean()
    std = cntArr.std()
    maxV = cntArr.max()
    minV = cntArr.min()    
    obj = {'mean':mean, 'square':square, 'std':std, 'max':maxV, 'min':minV}
    return obj

# 计算指定周的统计量
def statWeek(df, week):
    cntArr = df[df.week==week]['cnt'].values
    result = statCnt(cntArr)
    return result

# 添加one-hot编码并保留原字段
def addOneHot(df, colName):
    colDum = pd.get_dummies(df[colName], prefix=colName)
    df = pd.concat([df, colDum], axis=1)
    return df    

# 训练模型
def trainModel(X, y):
    clf = LinearRegression()
    clf.fit(X, y)
    print('Coefficients:', clf.coef_)
    return clf

# 划分训练集和测试集
def trainTestSplit(df, splitN, trainLabel):
    trainX = df[:splitN][trainLabel]
    trainY = df[:splitN]['cnt']
    testX = df[splitN:][trainLabel]
    testY = df[splitN:]['cnt']
    return (trainX, trainY, testX, testY)

# 导出预测结果
def exportResult(df, fileName):
    df.to_csv('./%s.txt' % fileName, sep='\t', header=False, index=False)


if __name__ == '__main__':
    # 导入数据
    df = importDf('../data/train_20171215.txt')

    # 特征提取
    startTime = datetime.now()
    df = pd.pivot_table(df,index=["date"], values=["cnt","day_of_week"], aggfunc={"cnt":np.sum, "day_of_week": np.max})
    df = tickWeek(df, 0)
    df.reset_index(inplace=True)
    df['day'] = df['week']*7 + df['day_of_week']
    df = addOneHot(df, 'day_of_week')
    df = df.dropna()
    print("feature time: ", datetime.now() - startTime)
    print(df.head(10))

    # 划分训练测试集
    splitN = int(df.index[-1] * 0.67)
    fea = ['day','day_of_week_1','day_of_week_2','day_of_week_3','day_of_week_4','day_of_week_5','day_of_week_6','day_of_week_7']
    trainX,trainY,testX,testY = trainTestSplit(df, splitN, fea)
    print(trainX.info())

    # 检验模型
    startTime = datetime.now()
    clf = trainModel(trainX.values, trainY.values)
    predictY = clf.predict(testX.values)
    cost = np.linalg.norm(predictY - testY.values)**2 / len(predictY)
    print("training time: ", datetime.now() - startTime)
    print("cost:", cost)

    # 正式模型
    modelName = "linear1"
    clf = trainModel(df[:][fea].values, df[:]['cnt'].values)
    joblib.dump(clf, './%s.pkl' % modelName, compress=3) 

    # 预测
    startTime = datetime.now()
    testDf = importDf('../data/test_A_20171225.txt')
    testDf = tickWeek(testDf, df.loc[df.index[-1], 'week'] + 1)
    testDf['day'] = testDf['week']*7 + testDf['day_of_week']
    dayDum = pd.get_dummies(testDf['day_of_week'], prefix='day_of_week')
    testDf = pd.concat([testDf, dayDum], axis=1)
    print(testDf.head())
    testDf['predict'] = clf.predict(testDf[fea].values)
    print(testDf[['date','predict']].head(10))
    exportResult(testDf[['date','predict']], "%s_A" % modelName)
