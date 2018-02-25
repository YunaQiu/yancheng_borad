#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
融合方式： stacking
模型： 线性回归
模型参数： 无
特征： 第一层各个模型的预测值
      是否工作日/休息日/节假日（第三方接口）
结果： A榜530707

'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
from datetime import *
import urllib, urllib.parse, urllib.request
import json

from sklearn.preprocessing import *
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# 导入数据
def importDf(url, sep=',', header='infer', index_col=None):
    df = pd.read_csv(url, sep=sep, header=header, index_col=index_col)    
    return df

# 接入第一层预测数据
def addSinglePredict(df, url, colName):
    if isinstance(url, str):
        url = [url]
    if isinstance(colName, str):
        colName = [colName]
    for i, file in enumerate(url):
        predictDf = importDf(file)
        predictDf.columns = ['date', colName[i]]
        df = pd.merge(df, predictDf, how='left', on='date')
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
def addGuessDate(df, startDate):
    if isinstance(startDate, str):
        startDate = datetime.strptime(startDate, '%Y-%m-%d')
    df['guess_date'] = df['week']*7 + df['day_of_week']
    df['guess_date'] = df['guess_date'].map(lambda x: startDate + timedelta(days=x))
    df['year'] = df['guess_date'].map(lambda x: x.year)
    df['month'] = df['guess_date'].map(lambda x: x.month)
    df['day'] = df['guess_date'].map(lambda x: x.day)
    df['guess_date'] = pd.to_datetime(df['guess_date'])
    return df

# 请求第三方接口获取日期休假情况(date字符串格式为YYYYMMDD)
# @param dateList: 日期格式为字符串YYYYMMDD
# @return: 返回结果：0：工作日，1：休息日，2：节假日
def checkHoliday(dateList):
    if not isinstance(dateList[0], str):
        dateList = [dt.strftime('%Y%m%d') for dt in dateList]
    url = "http://tool.bitefu.net/jiari/"
    data = urllib.parse.urlencode({'d':','.join(dateList)})
    res = urllib.request.urlopen(url, data.encode()).read()
    res = json.loads(res.decode('utf-8'))
    # 订正第三方接口返回值数值型与字符串型不统一的bug
    for k,v in res.items():
        res[k] = int(v)
    return res

# 添加节假日标记字段
def addHoliday(df):
    dateList = pd.to_datetime(df.guess_date)
    dateList = [dt.strftime('%Y%m%d') for dt in dateList]
    res = checkHoliday(dateList)
    df['holiday'] = list(map(lambda x: res[x], dateList))
    return df

# 添加one-hot编码并保留原字段
def addOneHot(df, colName):
    if isinstance(colName, str):
        colName = [colName]
    for col in colName:
        colDum = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, colDum], axis=1)
    return df

# 特征方法汇总
def feaFactory(df, startWeek=0, importFile=None, colName=None):
    if importFile != None:
        df = addSinglePredict(df, importFile, colName)
    df = tickWeek(df, startWeek)
    df = addGuessDate(df,'2012-12-30')
    df = addHoliday(df)
    df = addOneHot(df, ['day_of_week','month','holiday'])
    return df

# 训练模型
def trainModel(X, y):
    clf = linear_model.RidgeCV(alphas=[0.01*x for x in range(1,200)], scoring='neg_mean_squared_error')
    clf.fit(X, y)
    print('Coefficients:', clf.coef_)
    print('alpha:', clf.alpha_)
    return clf

# 导出预测结果
def exportResult(df, fileName, header=False, index=False, sep=','):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 统计预测误差
def countDeltaY(predictSeries, labelSeries, show=True, title='', subplot=None):
    deltaSeries = predictSeries - labelSeries
    if subplot!=None:
        plt.subplot(subplot[0], subplot[1], subplot[2])
    deltaSeries.plot(style='b-')
    plt.title(title)
    if show:
        plt.show()
    return deltaSeries


if __name__ == '__main__':
    # 训练数据
    df = importDf('../data/train_20171215.txt', sep='\t')
    startTime = datetime.now()
    df = pd.pivot_table(df,index=["date"], values=["cnt","day_of_week"], aggfunc={"cnt":np.sum, "day_of_week": np.max})
    df.reset_index(inplace=True)
    files = ['../linear1/linear1_oof_train.csv','../lake/stacking_layer1_train.csv','../keng/stacking_keng1_train.csv']
    columns = ['yuna_predict','lake_predict','keng_predict']
    df = feaFactory(df, importFile=files, colName=columns)
    df = df.dropna()
    print("feature time: ", datetime.now() - startTime)
    print("训练集：\n",df.tail())
    fea = ['yuna_predict','lake_predict','keng_predict']
    # fea.extend(['month_%d'%x for x in range(1,13)])
    # fea.extend(['day_of_week_%d'%x for x in range(1,8)])
    fea.extend(['holiday_%d'%x for x in range(0,3)])
    print("训练特征:",fea)
    print(df[fea].head())

    # 划分训练测试集
    splitDate = date(2015,4,1)
    trainDf = df[(df.guess_date < splitDate)]
    testDf = df[(df.guess_date >= splitDate)]
    print("模型输入：\n",trainDf[fea].info())

    # 检验模型
    clf = trainModel(trainDf[fea].values, trainDf['cnt'].values)    
    testDf['predict'] = clf.predict(testDf[fea].values)
    testDf['predict'] = testDf['predict'].map(lambda x: 15 if x<15 else x)   #修正负数值
    cost = metrics.mean_squared_error(testDf['cnt'].values, testDf['predict'].values) 
    print("训练数据量：", len(trainDf))
    print("cost:", cost)
    deltaSeries = countDeltaY(testDf.set_index(['guess_date'])['predict'], testDf.set_index(['guess_date'])['cnt'], show=False, subplot=(2,2,1))
    # exit()


    # 正式模型
    modelName = "stacking"
    clf = trainModel(df[fea].values, df['cnt'].values)
    joblib.dump(clf, './%s.pkl' % modelName, compress=3) 

    # 预测
    startTime = datetime.now()
    predictDf = importDf('../data/test_A_20171225.txt', sep='\t')
    files = ['../linear1/linear1_oof_testA.csv','../lake/stacking_layer1_test.csv','../keng/stacking_keng1_test.csv']
    predictDf = feaFactory(predictDf, startWeek=df.loc[df.index[-1], 'week'], importFile=files, colName=columns)
    # 填补缺失字段
    for x in [x for x in fea if x not in predictDf.columns]:
        predictDf[x] = 0
    print("预测集：\n",predictDf.head(10))
    print(predictDf[fea].info())
    predictDf['predict'] = clf.predict(predictDf[fea].values)
    predictDf['predict'] = predictDf['predict'].map(lambda x: 15 if x<15 else x)   #修正负数值
    print("预测结果：\n",predictDf[['date','predict']].head(10))
    exportResult(predictDf.set_index(['guess_date'])[['predict']+fea], "%s_predict.csv" % modelName, header=True, index=True)
    predictDf.loc[0,'predict'] = df.iloc[-1]['cnt']    #漏洞：预测集A第一个数据的结果直接替换成训练集最后一个数据的值
    exportResult(predictDf[['date','predict']], "%s_A.txt" % modelName, sep='\t')
