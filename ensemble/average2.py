#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
融合方式： 加权平均
权值计算：按照各个品牌的线下误差，作误差倒数的归一化加权
        其中yuna_predict的brand5数据不沿用，yuna_predict的brand8数据权重人为减小
后期处理：对节假日预测偏低的数据人为改动
结果： B榜（37048）

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
def importDf(url, sep='\t', header='infer', index_col=None):
    df = pd.read_csv(url, sep=sep, header=header, index_col=index_col)    
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
    df['day_of_year'] = df['guess_date'].map(lambda x: (x.date() - date(x.year,1,1)).days)
    df['month_day'] = df['guess_date'].map(lambda x: x.month*100+x.day)
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

# 特征方法汇总
def feaFactory(df, startWeek=0):
    df = tickWeek(df, startWeek)
    df = addGuessDate(df,'2012-12-30')
    df = addHoliday(df)
    return df

# 划分训练集和测试集
def trainTestSplit(df, splitN, trainLabel):
    trainX = df[:splitN][trainLabel]
    trainY = df[:splitN]['cnt']
    testX = df[splitN:][trainLabel]
    testY = df[splitN:]['cnt']
    return (trainX, trainY, testX, testY)

# 导出预测结果
def exportResult(df, fileName, header=False, index=False, sep='\t'):
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
    # 导入数据
    dfs = []
    dfs.append(importDf('../lake/resultB3_lake_aveHoliday2_dropData.txt', header=None, index_col=[0,1]))
    dfs.append(importDf('../keng/prophet_dateProperty_B_fusai_3_7.txt', header=None, index_col=[0,1]))
    dfs.append(importDf('../xgboost1/xgboost2B.txt', header=None, index_col=[0,1]))
    for df in dfs:
        df.columns = ['predict']
        df.index.names = ['date','brand']
    df = pd.concat(dfs, axis=1, keys=['lake', 'keng', 'yuna'])
    df.columns = [x[0]+'_'+x[1] for x in df.columns]
    df.reset_index(inplace=True)
    otherDf = importDf('../linear1/linear2B_predict.csv', sep=',')
    df = pd.merge(df, otherDf[['date','brand','guess_date','day_of_week','holiday']], how='left', on=['date','brand'])
    # 按排名加权平均
    rankList = {
        1:[0.24,0.35,0.41],
        2:[0.28,0.30,0.42],
        3:[0.40,0.26,0.34],
        4:[0.28,0.36,0.36],
        5:[0.50,0.50,0.00],
        6:[0.34,0.32,0.34],
        7:[0.36,0.36,0.28],
        8:[0.35,0.42,0.23],
        9:[0.35,0.37,0.28],
        10:[0.30,0.40,0.30]
    }
    for b in range(1,11):
        df.loc[df.brand==b,'predict'] = df.loc[df.brand==b,'lake_predict']*rankList[b][0] + df.loc[df.brand==b,'keng_predict']*rankList[b][1] + df.loc[df.brand==b,'yuna_predict']*rankList[b][2]
    exit()

    modelName = 'average2B_brand'
    exportResult(df, "%s_predict.csv" % modelName, header=True, sep=',')
    exportResult(df[['date','brand','predict']], "%s_0308.txt" % modelName)
