#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： xgboost
模型参数：'objective': 'reg:linear',
        'eval_metric':'rmse',
        'eta': 0.4,
        'gamma': 1400,
        'max_depth': 5
特征： 星期的one-hot特征
      真实日期的年/月/日,月日拼接的日期，月份的one-hot特征
      是否工作日/休息日/节假日（第三方接口）
      元旦后7天/春节前9天/春节后5天/国庆后5天的工作日序数
结果： A榜556770
遗留问题：本地验证集误差与线上误差差太远

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
import xgboost as xgb
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
def addGuessDate(df, startDate):
    if isinstance(startDate, str):
        startDate = datetime.strptime(startDate, '%Y-%m-%d')
    df['guess_date'] = df['week']*7 + df['day_of_week']
    df['guess_date'] = df['guess_date'].map(lambda x: startDate + timedelta(days=x))
    df['year'] = df['guess_date'].map(lambda x: x.year)
    df['month'] = df['guess_date'].map(lambda x: x.month)
    df['day'] = df['guess_date'].map(lambda x: x.day)
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

# 添加元旦后工作日字段
def addAfterNewyear(df, dayLen):
    df['is_after_newyear%d'%dayLen] = df['after_new_year_weight'] = 0
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-01-01'%y ,end='%d-01-30'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList
        dateList = dateSeries[dateSeries==0].index[:dayLen]
        # df.loc[df.guess_date.isin(dateList), 'is_after_newyear%d'%dayLen] = 1

        weightSeries = pd.Series([dayLen-i for i in range(0,dayLen)], index=dateList)
        interIndex = np.intersect1d(weightSeries.index, df.guess_date)
        df.loc[df.guess_date.isin(dateList), 'after_new_year_weight'] = weightSeries[interIndex].values
    return df

# 添加春节前后工作日字段
def addBeforeSpringFest(df, beforeDayLen):
    df['is_before_spring_fest%d'%beforeDayLen] = df['before_spring_fest_weight'] = 0
    springFest = {
        2013:date(2013,2,10),
        2014:date(2014,1,31),
        2015:date(2015,2,19),
        2016:date(2016,2,8),
        2017:date(2017,1,28)}
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-01-01'%y ,end='%d-03-10'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList
        
        beforeList = dateSeries[:springFest[y]][dateSeries==0].index[-beforeDayLen:]
        # df.loc[df.guess_date.isin(beforeList), 'is_before_spring_fest%d'%beforeDayLen] = 1
        weightSeries = pd.Series(list(range(1,beforeDayLen+1)), index=beforeList)
        interIndex = np.intersect1d(weightSeries.index, df.guess_date)
        df.loc[df.guess_date.isin(beforeList), 'before_spring_fest_weight'] = weightSeries[interIndex].values
    return df

# 添加春节前后工作日字段
def addAfterSpringFest(df, afterDayLen):
    df['is_after_spring_fest%d'%afterDayLen] = df['after_spring_fest_weight'] = 0
    springFest = {
        2013:date(2013,2,10),
        2014:date(2014,1,31),
        2015:date(2015,2,19),
        2016:date(2016,2,8),
        2017:date(2017,1,28)}
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-01-01'%y ,end='%d-03-10'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList

        afterList = dateSeries[springFest[y]:][dateSeries==0].index[:afterDayLen]
        # df.loc[df.guess_date.isin(afterList), 'is_after_spring_fest%d'%afterDayLen] = 1
        weightSeries = pd.Series(list(range(afterDayLen,0,-1)), index=afterList)
        interIndex = np.intersect1d(weightSeries.index, df.guess_date)
        df.loc[df.guess_date.isin(afterList), 'after_spring_fest_weight'] = weightSeries[interIndex].values
    return df

# 添加国庆后工作日字段
def addAfterNational(df, dayLen):
    df['is_after_national%d'%dayLen] = df['after_national_weight'] = 0
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-10-01'%y ,end='%d-10-30'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList
        dateList = dateSeries[dateSeries==0].index[:dayLen]
        # df.loc[df.guess_date.isin(dateList), 'is_after_national%d'%dayLen] = 1
        weightSeries = pd.Series(list(range(dayLen,0,-1)), index=dateList)
        interIndex = np.intersect1d(weightSeries.index, df.guess_date)
        df.loc[df.guess_date.isin(dateList), 'after_national_weight'] = weightSeries[interIndex].values
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

# 特征方法汇总
def feaFactory(df, startWeek=0):
    df = tickWeek(df, startWeek)
    df = addGuessDate(df,'2012-12-30')
    df = addHoliday(df)
    df = addAfterNewyear(df, 7)
    df = addBeforeSpringFest(df, 9)
    df = addAfterSpringFest(df, 5)
    df = addAfterNational(df, 5)
    df = addOneHot(df, ['day_of_week','month','holiday'])
    return df

# 训练模型
def trainModel(X, y, feaNames):
    dtrain = xgb.DMatrix(X, label=y)
    param = {
        'objective': 'reg:linear',
        'eval_metric':'rmse',
        'eta': 0.4,
        'gamma': 1400,
        'max_depth': 5}
    clf = xgb.train(param.items(), dtrain, num_boost_round=8)
    # clf = xgb.XGBRegressor(max_depth=4, learning_rate=0.4, eval_metric='rmse')
    # clf = clf.fit(X, y, eval_metric='rmse')

    fscore=clf.get_score()
    scoreDf = pd.DataFrame(feaNames, columns=['fea'])
    scoreDf['importance'] = np.nan
    for k,v in fscore.items():
        scoreDf.loc[int(k[1:]), 'importance'] = v
    print(scoreDf.dropna().sort_index(by=['importance'], ascending=False))
    return clf

# 划分训练集和测试集
def trainTestSplit(df, splitN, trainLabel):
    trainX = df[:splitN][trainLabel]
    trainY = df[:splitN]['cnt']
    testX = df[splitN:][trainLabel]
    testY = df[splitN:]['cnt']
    return (trainX, trainY, testX, testY)

# 导出预测结果
def exportResult(df, fileName, header=False, index=False):
    df.to_csv('./%s.txt' % fileName, sep='\t', header=header, index=index)

# 统计预测误差
def countDeltaY(predictSeries, labelSeries):
    deltaSeries = predictSeries - labelSeries
    deltaSeries.plot(style='b-')
    plt.show()
    return deltaSeries


if __name__ == '__main__':
    # 导入数据
    df = importDf('../data/train_20171215.txt')

    # 特征提取
    startTime = datetime.now()
    df = pd.pivot_table(df,index=["date"], values=["cnt","day_of_week"], aggfunc={"cnt":np.sum, "day_of_week": np.max})
    df.reset_index(inplace=True)
    df = feaFactory(df)
    scaleCols = 'year'
    df,scaler = scalerFea(df, scaleCols)
    df = df.dropna()
    print("feature time: ", datetime.now() - startTime)
    print("训练集：\n",df.tail())
    fea = ['year','month','day','month_day',
        'after_new_year_weight',
        'before_spring_fest_weight','after_spring_fest_weight',
        'after_national_weight']
    fea.extend(['month_%d'%x for x in range(1,13)])
    fea.extend(['day_of_week_%d'%x for x in range(1,8)])
    fea.extend(['holiday_%d'%x for x in range(0,3)])
    print("训练特征:",fea)

    # 划分训练测试集
    splitN = int(df.index[-1] * 0.67)
    trainDf = df.loc[:splitN]
    testDf = df.loc[splitN:]
    # splitDate = date(2015,4,1)
    # trainDf = df[(df.guess_date < splitDate)]
    # testDf = df[(df.guess_date >= splitDate)]
    print("模型输入：\n",trainDf[fea].info())

    # 检验模型
    startTime = datetime.now()
    clf = trainModel(trainDf[fea].values, trainDf['cnt'].values, fea)
    # testDf['predict'] = clf.predict(testDf[fea].values)
    testDf['predict'] = clf.predict(xgb.DMatrix(testDf[fea].values))
    cost = metrics.mean_squared_error(testDf['cnt'].values, testDf['predict'].values) 
    print("training time: ", datetime.now() - startTime)
    # print("训练数据量：", trainN)
    print("cost:", cost)
    deltaSeries = countDeltaY(testDf.set_index(['guess_date'])['predict'], testDf.set_index(['guess_date'])['cnt'])
    # print(deltaSeries[abs(deltaSeries)>1000])
    # exit()

    # 正式模型
    modelName = "xgboost1"
    # trainDf = df[df.guess_date >= df.iloc[-1].guess_date-trainN]
    clf = trainModel(df[fea].values, df['cnt'].values, fea)
    # joblib.dump(clf, './%s.pkl' % modelName, compress=3) 

    # 预测
    startTime = datetime.now()
    predictDf = importDf('../data/test_A_20171225.txt')
    predictDf = feaFactory(predictDf, startWeek=df.loc[df.index[-1], 'week'])
    # 填补缺失字段
    for x in range(1,13):
        if 'month_%d'%x not in predictDf.columns:
            predictDf['month_%d'%x] = 0
    predictDf[scaleCols] = scaler.transform(predictDf[scaleCols].values)
    print("预测集：\n",predictDf.head(10))
    print(predictDf[fea].info())
    predictDf['predict'] = clf.predict(xgb.DMatrix(predictDf[fea].values))
    print("预测结果：\n",predictDf[['date','predict']].head(10))
    # exportResult(predictDf.set_index(['guess_date'])[['predict']+fea], "%s_predict" % modelName, header=True, index=True)
    predictDf.loc[0,'predict'] = df.iloc[-1]['cnt']    #漏洞：预测集A第一个数据的结果直接替换成训练集最后一个数据的值
    # exportResult(predictDf[['date','predict']], "%s_A" % modelName)
