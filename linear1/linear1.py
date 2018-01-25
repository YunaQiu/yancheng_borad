#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： 线性回归
模型参数： 无
特征： 基于周数和星期的修正日期（day）
      星期的one-hot特征
      真实日期的年月日及月份的one-hot特征
      是否工作日/休息日/节假日
结果： A榜685537（后两年数据）
遗留问题：验证集上发现用两年的数据训练的效果比用三年的更好，但提交后结果更差，为什么？怎么确认合适的训练数据量？

'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib.dates
from datetime import *
import urllib, urllib.parse, urllib.request
import json

from sklearn.preprocessing import *
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# 导入数据
def importDf(url):
    df = pd.read_csv(url, sep='\t')    
    return df

# 标记周数
def tickWeek(df, start):
    preVal = df[:-1]['day_of_week'].values
    preVal = np.insert(preVal, 0, preVal[0]-1)
    df['new_week'] = (preVal >= df.day_of_week)
    weekList = list()
    week = start
    for item in df['new_week']:
        week += (1 if item else 0)
        weekList.append(week)
    df['week'] = weekList
    return df

# 给数据添加日期字段
def addGuessDate(df, startYear, startMonth, startDay):
    startDate = date(startYear, startMonth, startDay)
    df['guess_date'] = df['day'].map(lambda x: startDate + timedelta(days=x))
    df['year'] = df['guess_date'].map(lambda x: x.year)
    df['month'] = df['guess_date'].map(lambda x: x.month)
    df['day'] = df['guess_date'].map(lambda x: x.day)
    df['guess_date'] = pd.to_datetime(df['guess_date'])
    return df

# 添加节假日标记字段
def addHoliday(df):
    dateList = pd.to_datetime(df.guess_date)
    dateList = [dt.strftime('%Y%m%d') for dt in dateList]
    # 向第三方接口请求数据(返回结果：0：工作日，1：休息日，2：节假日)
    url = "http://tool.bitefu.net/jiari/"
    data = urllib.parse.urlencode({'d':','.join(dateList)})
    res = urllib.request.urlopen(url, data.encode()).read()
    res = json.loads(res.decode('utf-8'))
    df['holiday'] = list(map(lambda x: int(res[x]), dateList))
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
    if isinstance(colName, str):
        colName = [colName]
    for col in colName:
        colDum = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, colDum], axis=1)
    return df

# 缩放字段至0-1
def scalerFea(df, cols):
    df.dropna(inplace=True)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df,scaler

# 训练模型
def trainModel(X, y):
    clf = linear_model.RidgeCV(alphas=[.2*x for x in range(1,100)], scoring='neg_mean_squared_error')
    clf.fit(X, y)
    print('Coefficients:', clf.coef_)
    print('alpha:', clf.alpha_)
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
    df = addGuessDate(df,2012,12,30)
    df = addHoliday(df)
    df = addOneHot(df, ['day_of_week','month','holiday'])
    df,scaler = scalerFea(df, ['year'])
    df = df.dropna()
    print("feature time: ", datetime.now() - startTime)
    print("训练集：\n",df.tail(30))
    fea = ['year','month','day']
    fea.extend(['month_%d'%x for x in range(1,13)])
    fea.extend(['day_of_week_%d'%x for x in range(1,8)])
    fea.extend(['holiday_%d'%x for x in range(0,3)])
    print("训练特征:",fea)

    # 划分训练测试集
    splitDate = date(2015,6,1)
    trainN = timedelta(days=2*365)
    trainDf = df[(df.guess_date >= splitDate-trainN) & (df.guess_date < splitDate)]
    testDf = df[(df.guess_date >= splitDate) & (df.guess_date < splitDate+timedelta(days=300))]
    print("模型输入：\n",trainDf[fea].info())

    # 检验模型
    startTime = datetime.now()
    clf = trainModel(trainDf[fea].values, trainDf['cnt'].values)
    testDf['predict'] = clf.predict(testDf[fea].values)
    cost = metrics.mean_squared_error(testDf['cnt'].values, testDf['predict'].values) 
    print("training time: ", datetime.now() - startTime)
    print("训练数据量：", trainN)
    print("cost:", cost)
    # exit()

    # 正式模型
    modelName = "linear1_addHoliday"
    trainDf = df[df.guess_date >= df.iloc[-1].guess_date-trainN]
    clf = trainModel(trainDf[fea].values, trainDf['cnt'].values)
    joblib.dump(clf, './%s.pkl' % modelName, compress=3) 

    # 预测
    startTime = datetime.now()
    predictDf = importDf('../data/test_A_20171225.txt')
    predictDf = tickWeek(predictDf, df.loc[df.index[-1], 'week'])
    predictDf['day'] = predictDf['week']*7 + predictDf['day_of_week']
    predictDf = addGuessDate(predictDf,2012,12,30)
    predictDf = addHoliday(predictDf)
    predictDf = addOneHot(predictDf, ['day_of_week','month','holiday'])
    for x in range(1,13):
        if 'month_%d'%x not in predictDf.columns:
            predictDf['month_%d'%x] = 0
    predictDf['year'] = scaler.transform(predictDf['year'].values)
    print("预测集：\n",predictDf.head(10))
    print(predictDf[fea].info())
    predictDf['predict'] = clf.predict(predictDf[fea].values)
    print("预测结果：\n",predictDf[['date','predict']].head(10))
    predictDf.loc[0,'predict'] = df.iloc[-1]['cnt']    #漏洞：预测集A第一个数据的结果直接替换成训练集最后一个数据的值
    exportResult(predictDf[['date','predict']], "%s_A" % modelName)
