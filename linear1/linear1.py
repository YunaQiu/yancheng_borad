#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： 线性回归
模型参数： 无
特征： 星期1/2/3/4/6的onehot标记
      月份1/2/6/7/8/11/12的onehot标记
      节假日类型0/1/2的onehot标记（第三方接口）
      本月销售量（取自零售量比赛数据，其中17年11月数据根据网上的16年同期增长比估算）
      元旦后5个工作日，元旦后工作日的修正权重
      春节前9个工作日/春节前5个工作日/春节前1个工作日/春节后3个工作日
      五一后1个工作日
      国庆后1个工作日
结果： A榜约42w
      B榜未知（融合后41w）

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
from sklearn.model_selection import train_test_split, KFold
from sklearn.externals import joblib

# 导入数据
def importDf(url, sep='\t', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, header=header, index_col=index_col, names=colNames)
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
    df['year_month'] = df['guess_date'].map(lambda x: x.year*100+x.month)
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

# 添加周日休息日标记
def addSundayHoliday(df):
    df['is_sunday_holiday'] = df.apply(lambda x: 1 if (x.day_of_week==7)&(x.holiday>0) else 0, axis=1)
    return df

# 添加元旦前工作日字段
def addBeforeNewyear(df, dayLen):
    df['is_before_newyear%d'%dayLen] = df['before_new_year_weight'] = 0
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-12-01'%(y-1) ,end='%d-01-01'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList
        dateList = dateSeries[dateSeries==0].index[-dayLen:]
        df.loc[df.guess_date.isin(dateList), 'is_before_newyear%d'%dayLen] = 1

        weightSeries = pd.Series([dayLen-i for i in range(0,dayLen)], index=dateList)
        interIndex = np.intersect1d(weightSeries.index, df.guess_date)
        df.loc[df.guess_date.isin(dateList), 'before_new_year_weight'] = weightSeries[interIndex].values
    return df

# 添加元旦后工作日字段
def addAfterNewyear(df, dayLen):
    df['is_after_newyear%d'%dayLen] = df['after_new_year_weight'] = 0
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-01-01'%y ,end='%d-01-30'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList
        dateList = dateSeries[dateSeries==0].index[:dayLen]
        df.loc[df.guess_date.isin(dateList), 'is_after_newyear%d'%dayLen] = 1

        weightSeries = pd.Series([dayLen-i for i in range(0,dayLen)], index=dateList)
        interIndex = np.intersect1d(weightSeries.index, df.guess_date)
        df.loc[df.guess_date.isin(dateList), 'after_new_year_weight'] = weightSeries[interIndex].values
    return df

# 添加春节前工作日字段
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
        df.loc[df.guess_date.isin(beforeList), 'is_before_spring_fest%d'%beforeDayLen] = 1
        weightSeries = pd.Series(list(range(1,beforeDayLen+1)), index=beforeList)
        interIndex = np.intersect1d(weightSeries.index, df.guess_date)
        df.loc[df.guess_date.isin(beforeList), 'before_spring_fest_weight'] = weightSeries[interIndex].values
    return df

# 添加春节后工作日字段
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
        df.loc[df.guess_date.isin(afterList), 'is_after_spring_fest%d'%afterDayLen] = 1
        weightSeries = pd.Series(list(range(afterDayLen,0,-1)), index=afterList)
        interIndex = np.intersect1d(weightSeries.index, df.guess_date)
        df.loc[df.guess_date.isin(afterList), 'after_spring_fest_weight'] = weightSeries[interIndex].values
    return df

# 添加五一后工作日字段
def addAfterWorker(df, dayLen):
    df['is_after_worker%d'%dayLen] = 0
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-05-01'%y ,end='%d-05-10'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList
        dateList = dateSeries[dateSeries==0].index[:dayLen]
        df.loc[df.guess_date.isin(dateList), 'is_after_worker%d'%dayLen] = 1
    return df

# 添加国庆前工作日字段
def addBeforeNational(df, dayLen):
    df['is_before_national%d'%dayLen] = 0
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-09-01'%y ,end='%d-10-01'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList
        dateList = dateSeries[dateSeries==0].index[-dayLen:]
        df.loc[df.guess_date.isin(dateList), 'is_before_national%d'%dayLen] = 1
    return df

# 添加国庆后工作日字段
def addAfterNational(df, dayLen):
    df['is_after_national%d'%dayLen] = 0
    for y in df.year.value_counts().index:
        dateList = pd.date_range(start='%d-10-01'%y ,end='%d-10-10'%y, freq='D')
        dateSeries = pd.Series(checkHoliday(dateList.strftime('%Y%m%d')))
        dateSeries.index = dateList
        dateList = dateSeries[dateSeries==0].index[:dayLen]
        df.loc[df.guess_date.isin(dateList), 'is_after_national%d'%dayLen] = 1
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

# 添加销售量数据
def addSaleFea(df):
    saleDf = importDf('../extra_data/yancheng_sale.csv', sep=',')
    saleDf = pd.pivot_table(saleDf, index=['sale_date'], values='sale_quantity', aggfunc=np.sum)
    saleDf.loc[201711, 'sale_quantity'] = saleDf.loc[201611, 'sale_quantity'] * 1.025
    saleDf.columns = ['this_month_sale']
    saleDf['last_month_sale'] = saleDf['last_2month_sale'] = np.nan
    saleDf.loc[saleDf.index[1]:,'last_month_sale'] = saleDf['this_month_sale'].values[:-1]
    saleDf.loc[saleDf.index[1]:,'last_2month_sale'] = saleDf['last_month_sale'].values[:-1]
    saleDf['this_month_sale_growth'] = (saleDf['this_month_sale'].values - saleDf['last_month_sale'].values) / saleDf['last_month_sale'].values
    saleDf['last_month_sale_growth'] = (saleDf['last_month_sale'].values - saleDf['last_2month_sale'].values) / saleDf['last_2month_sale'].values
    df = pd.merge(df, saleDf, how='left', left_on='year_month', right_index=True)
    return df

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
    df = addSundayHoliday(df)
    df = addBeforeNewyear(df, 3)
    df = addAfterNewyear(df, 5)
    df = addBeforeSpringFest(df, 1)
    df = addBeforeSpringFest(df, 9)
    df = addBeforeSpringFest(df, 5)
    df = addAfterSpringFest(df, 5)
    df = addAfterSpringFest(df, 3)
    df = addAfterWorker(df, 1)
    df = addBeforeNational(df, 3)
    df = addAfterNational(df, 1)
    df = addSaleFea(df)
    df = addOneHot(df, ['day_of_week','year','month','holiday'])
    return df

# 训练模型
def trainModel(X, y):
    clf = linear_model.RidgeCV(alphas=[0.01*x for x in range(1,200)], scoring='neg_mean_squared_error')
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
def exportResult(df, fileName, header=False, index=False, sep=','):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 统计预测误差
def countDeltaY(predictSeries, labelSeries, show=True, title='', subplot=None):
    deltaSeries = predictSeries - labelSeries
    if subplot!=None:
        plt.subplot(subplot[0], subplot[1], subplot[2])
    fig, ax = plt.subplots(figsize=(15,6))
    deltaSeries.plot(ax=ax,style='b-')
    plt.title(title)
    if show:
        plt.show()
    return deltaSeries

def getOof(clf, trainX, trainY, testX, nFold=10):
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    kf = KFold(n_splits=nFold, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        clf.fit(kfTrainX, kfTrainY)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
    oofTest[:] = oofTestSkf.mean(axis=1)
    return oofTrain, oofTest


if __name__ == '__main__':
    # 导入数据
    df = importDf('../data/train_20171215.txt')
    dfA = importDf('../data/test_A_20171225.txt')
    answerDfA = importDf('../data/answer_A_20180225.txt', header=None, colNames=['date','cnt'])
    dfA['brand'] = 0
    dfA = dfA.merge(answerDfA, how='left', on='date')
    df = pd.concat([df, dfA], ignore_index=True)  

    # 特征提取
    startTime = datetime.now()
    df = pd.pivot_table(df,index=["date"], values=["cnt","day_of_week"], aggfunc={"cnt":np.sum, "day_of_week": np.max})
    df.reset_index(inplace=True)
    df = feaFactory(df)
    scaleCols = ['year','last_month_sale','this_month_sale']
    df,scaler = scalerFea(df, scaleCols)
    df = df.dropna()
    print("feature time: ", datetime.now() - startTime)
    print("训练集：\n",df.tail())
    fea = [
        'this_month_sale',#'last_month_sale',
        'is_after_newyear5','after_new_year_weight',
        'is_before_spring_fest1','is_before_spring_fest5','is_before_spring_fest9','is_after_spring_fest3',
        'is_after_worker1',
        'is_after_national1'
        ]
    fea.extend(['month_%d'%x for x in [1,2,6,7,8,11,12]])
    fea.extend(['day_of_week_%d'%x for x in [1,2,3,4,6]])
    fea.extend(['holiday_%d'%x for x in range(0,3)])
    # 填补缺失字段
    for x in [x for x in fea if x not in df.columns]:
        df[x] = 0
    print("训练特征:",fea)

    # 用滑动窗口检验模型
    feaDf = pd.DataFrame(index=fea+['cost'])
    for dt in pd.date_range(start='2016-02-01', end='2016-04-01', freq='MS'):
        splitDate = dt
        trainDf = df[(df.guess_date < splitDate)]
        testDf = df[(df.guess_date >= splitDate) & (df.guess_date < splitDate+timedelta(days=365))]
        print("模型输入：\n",trainDf[fea].info())

        # 检验模型
        startTime = datetime.now()
        clf = trainModel(trainDf[fea].values, trainDf['cnt'].values)
        testDf['predict'] = clf.predict(testDf[fea].values)
        testDf.loc[:,'predict'] = testDf['predict'].map(lambda x: 15 if x<15 else x)   #修正负数值
        testDf.loc[testDf.is_sunday_holiday==1,'predict'] = testDf.loc[testDf.is_sunday_holiday==1,'predict'].map(lambda x: 100 if x>100 else x)   #修正周日休息日的数据
        cost = metrics.mean_squared_error(testDf['cnt'].values, testDf['predict'].values) 
        print("training time: ", datetime.now() - startTime)
        feaDf[dt] = list(clf.coef_)+[cost]
        # deltaSeries = countDeltaY(testDf.set_index(['guess_date'])['predict'], testDf.set_index(['guess_date'])['cnt'], show=False)
    print(feaDf)

    # 导出完整训练集预测记录
    df['predict'], temp = getOof(clf, df[fea].values, df['cnt'].values, testDf[fea].values, nFold=10)
    df['delta'] = df['predict'] - df['cnt']
    deltaSeries = countDeltaY(df.set_index(['guess_date'])['predict'], df.set_index(['guess_date'])['cnt'], show=False)
    exportResult(df.set_index(['date'])[['guess_date','cnt','predict','delta','day_of_week','holiday','this_month_sale']], "train_predict.csv", header=True, index=True)
    # exit()


    # 正式模型
    modelName = "linear1B"
    # trainDf = df[df.guess_date >= df.iloc[-1].guess_date-trainN]
    clf = trainModel(df[fea].values, df['cnt'].values)
    joblib.dump(clf, './%s.pkl' % modelName, compress=3) 

    # 预测
    startTime = datetime.now()
    predictDf = importDf('../data/test_B_20171225.txt')
    predictDf = feaFactory(predictDf, startWeek=df.loc[df.index[-1], 'week'])
    # 填补缺失字段
    for x in [x for x in fea if x not in predictDf.columns]:
        predictDf[x] = 0
    predictDf[scaleCols] = scaler.transform(predictDf[scaleCols].values)
    print("预测集：\n",predictDf.head())
    print(predictDf[fea].info())
    predictDf['predict'] = clf.predict(predictDf[fea].values)
    print("预测结果：\n",predictDf[['date','predict']].head(10))
    exportResult(predictDf.set_index(['date'])[['guess_date','predict','day_of_week','holiday','this_month_sale']], "%s_predict.csv" % modelName, header=True, index=True)
    predictDf['predict'] = predictDf['predict'].map(lambda x: 15 if x<15 else x)   #修正负数值
    predictDf.loc[predictDf.is_sunday_holiday==1,'predict'] = predictDf.loc[predictDf.is_sunday_holiday==1,'predict'].map(lambda x: 100 if x>100 else x)   #修正周日休息日的数据
    exportResult(predictDf[['date','predict']], "%s.txt" % modelName, sep='\t')

    # 生成模型融合数据集
    df['predict'], predictDf['predict'] = getOof(clf, df[fea].values, df['cnt'].values, predictDf[fea].values, nFold=10)
    predictDf['predict'] = predictDf['predict'].map(lambda x: 15 if x<15 else x)   #修正负数值
    exportResult(df[['date','predict']], "%s_oof_train2.csv" % modelName, header=True)
    exportResult(predictDf[['date','predict']], "%s_oof_testB.csv" % modelName, header=True)

