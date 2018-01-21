#!/usr/bin/env 
# coding: utf-8

'''
模型： 线性回归
模型参数： 无
特征： 基于周数和星期的修正日期（day）
      星期的one-hot特征
      上一周的均值方差统计量
备注： 预测时按周分批预测
结果： A榜978764

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

# 计算均方值
def meanSquare(npArr):
    return np.square(npArr).mean()

# 计算统计量
def statCnt(cntArr):
    cntArr = np.array(cntArr)
    mean = cntArr.mean()
    # square = meanSquare(cntArr)
    std = cntArr.std()
    maxV = cntArr.max()
    minV = cntArr.min()    
    result = [mean, std, maxV, minV]
    return result

# 添加过去第i周的统计量
def statWeek(df, weeks):
    if isinstance(weeks, int):
        weeks = [weeks]
    colName = []
    for i in weeks:
        weekDf = pd.pivot_table(df, index=['week'], values=['cnt'], aggfunc=[np.mean, np.std, np.max, np.min])
        weekDf.columns = ['mean%d'%i, 'std%d'%i, 'max%d'%i, 'min%d'%i]
        colName.extend(weekDf.columns)
        weekDf.index += i
        df = pd.merge(df, weekDf, left_on='week', right_index=True, how='left')
    return df,colName

# 添加one-hot编码并保留原字段
def addOneHot(df, colName):
    colDum = pd.get_dummies(df[colName], prefix=colName)
    df = pd.concat([df, colDum], axis=1)
    return df    

# 缩放字段至0-1
def scalerFea(df, cols):
    df.dropna(inplace=True)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df,scaler.scale_

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

# 检验模型
def validModel(trainX, trainY, testX, testY):
    clf = trainModel(trainX, trainY)
    predictY = clf.predict(testX)
    cost = np.linalg.norm(predictY - testY)**2 / len(predictY)
    print("cost:", cost)
    
# 预测数据集
def predict(trainDf, testDf, fea, scaler, statCols):
    predWeekList = testDf['week'].drop_duplicates().values
    for x in statCols:
        testDf[x] = np.nan
    for w in predWeekList:
        if w==predWeekList[0]:
            stat = statCnt(trainDf.loc[trainDf.week==(w-1),'cnt'].values)
        else:
            stat = statCnt(testDf.loc[testDf.week==(w-1),'predict'].values)
        stat *= scaler
        testDf.loc[testDf.week==w, statCols] = stat
        testDf.loc[testDf.week==w, 'predict'] = clf.predict(testDf.loc[testDf.week==w, fea].values)
    return testDf

# 导出预测结果
def exportResult(df, fileName):
    df.to_csv('./%s.txt' % fileName, sep='\t', header=False, index=False)


if __name__ == "__main__":
    # 导入数据
    df = importDf('./data/train_20171215.txt')

    # 特征提取
    startTime = datetime.now()
    df = pd.pivot_table(df,index=["date"], values=["cnt","day_of_week"], aggfunc={"cnt":np.sum, "day_of_week": np.max})
    df = tickWeek(df, 0)
    df.reset_index(inplace=True)
    df['day'] = df['week']*7 + df['day_of_week']
    df, statCols = statWeek(df, 1)
    df = addOneHot(df, 'day_of_week')
    # statCols = ['mean1','std1','max1','min1']
    df, scaler = scalerFea(df, statCols)
    print("stat scaler: ", scaler)
    df = df.dropna()
    print("feature time: ", datetime.now() - startTime)
    print(df.info())

    # 划分训练测试集
    splitN = int(df.index[-1] * 0.67)
    fea = ['day','day_of_week_1','day_of_week_2','day_of_week_3','day_of_week_4','day_of_week_5','day_of_week_6','day_of_week_7',
        'mean1','std1']
    # trainDf = df[:splitN]
    # testDf = df[splitN:]
    # trainX,trainY,testX,testY = trainTestSplit(df, splitN, fea)
    # print(trainX.info())

    # # 一周周预测
    # startTime = datetime.now()
    # clf = trainModel(trainX, trainY)
    # testDf = predict(trainDf, testDf, fea, scaler, statCols)
    # print("predict time: ", datetime.now() - startTime)

    # cost = np.linalg.norm(testDf['predict'].values - testY)**2 / len(testY)
    # print("cost:", cost)

    # 检验模型
    # validModel(trainX.values, trainY.values, testX.values, testY.values)

    # 正式模型
    modelName = "linear2"
    clf = trainModel(df[-750:][fea].values, df[-750:]['cnt'].values)
    joblib.dump(clf, './%s.pkl' % modelName, compress=3) 

    # exit()
    # 预测
    startTime = datetime.now()
    predictDf = importDf('./data/test_A_20171225.txt')
    predictDf = tickWeek(predictDf, df.loc[df.index[-1], 'week'] + 1)
    predictDf['day'] = predictDf['week']*7 + predictDf['day_of_week']
    predictDf = addOneHot(predictDf, 'day_of_week')
    print(predictDf.head())

    # testInput = predictDf.drop(['date','week'], axis=1)
    # print(testInput.info())
    # predictDf['predict'] = clf.predict(predictDf[fea].values)
    predictDf = predict(df, predictDf, fea, scaler, statCols)
    print(predictDf.info())
    exportResult(predictDf[['date','predict']], "%s_A" % modelName)
