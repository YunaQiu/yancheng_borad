#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： 岭回归CV，按品牌拆分成10个模型训练
模型参数： alpha取值：0.02:0.02:10
特征： 星期1~6的onehot标记
      月份1~12的onehot标记
      节假日类型0/1/2的onehot标记
      周日休息日标记
      本月销售量（取自零售量比赛数据，其中17年11月数据根据网上的16年同期增长比估算）
      元旦后3个工作日，元旦后5个工作日
      春节前9个工作日/春节前5个工作日/春节前1个工作日/春节后3个工作日
      五一后1个工作日
      国庆后1个工作日
后期处理：小于0的值设为0,周日休息日大于100的值减100
结果： A榜（34835）

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
    tempDf = df.drop_duplicates(['date'])
    preVal = tempDf[:-1]['day_of_week'].values
    preVal = np.insert(preVal, 0, preVal[0]-1)
    tempDf['new_week'] = (preVal >= tempDf.day_of_week)
    weekList = list()
    week = start
    for item in tempDf['new_week']:
        week += (1 if item else 0)
        weekList.append(week)
    tempDf['week'] = weekList
    df = df.merge(tempDf[['date','week','new_week']], how='left', on='date')
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

# 获取日期休假情况(date字符串格式为YYYY-MM-DD)
def getHolidayDf():
    df = importDf('../extra_data/holiday.csv', sep=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

# 添加节假日标记字段
def addHoliday(df):
    holidayDf = getHolidayDf()
    df['holiday'] = df.guess_date.map(lambda x: holidayDf.loc[x, 'holiday'])
    return df

# 添加周日休息日标记
def addSundayHoliday(df):
    df['is_sunday_holiday'] = df.apply(lambda x: 1 if (x.day_of_week==7)&(x.holiday>0) else 0, axis=1)
    return df

# 添加元旦前工作日字段
def addBeforeNewyear(df, dayLen):
    df['is_before_newyear%d'%dayLen] = df['before_new_year_weight'] = 0
    for y in df.year.value_counts().index:
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-12-01'%(y-1)):('%d-01-01'%y)][holidayDf.holiday==0].index[-dayLen:]
        df.loc[df.guess_date.isin(dateList), 'is_before_newyear%d'%dayLen] = 1

        weightSeries = pd.Series([dayLen-i for i in range(0,dayLen)], index=dateList)
        df.loc[df.guess_date.isin(dateList), 'before_new_year_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
    return df

# 添加元旦后工作日字段
def addAfterNewyear(df, dayLen):
    df['is_after_newyear%d'%dayLen] = df['after_new_year_weight'] = 0
    for y in df.year.value_counts().index:
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-01-01'%y):('%d-01-30'%y)][holidayDf.holiday==0].index[:dayLen]
        df.loc[df.guess_date.isin(dateList), 'is_after_newyear%d'%dayLen] = 1

        weightSeries = pd.Series([dayLen-i for i in range(0,dayLen)], index=dateList)
        df.loc[df.guess_date.isin(dateList), 'after_new_year_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
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
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-01-01'%y):(springFest[y])][holidayDf.holiday==0].index[-beforeDayLen:]
        df.loc[df.guess_date.isin(dateList), 'is_before_spring_fest%d'%beforeDayLen] = 1
        weightSeries = pd.Series(list(range(1,beforeDayLen+1)), index=dateList)
        df.loc[df.guess_date.isin(dateList), 'before_spring_fest_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
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
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[(springFest[y]):('%d-03-10'%y)][holidayDf.holiday==0].index[:afterDayLen]
        df.loc[df.guess_date.isin(dateList), 'is_after_spring_fest%d'%afterDayLen] = 1
        weightSeries = pd.Series(list(range(afterDayLen,0,-1)), index=dateList)
        df.loc[df.guess_date.isin(dateList), 'after_spring_fest_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
    return df

# 添加五一后工作日字段
def addAfterWorker(df, dayLen):
    df['is_after_worker%d'%dayLen] = 0
    for y in df.year.value_counts().index:
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-05-01'%y):('%d-05-10'%y)][holidayDf.holiday==0].index[:dayLen]
        df.loc[df.guess_date.isin(dateList), 'is_after_worker%d'%dayLen] = 1
    return df

# 添加国庆前工作日字段
def addBeforeNational(df, dayLen):
    df['is_before_national%d'%dayLen] = 0
    for y in df.year.value_counts().index:
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-09-01'%y):('%d-10-01'%y)][holidayDf.holiday==0].index[-dayLen:]
        df.loc[df.guess_date.isin(dateList), 'is_before_national%d'%dayLen] = 1
    return df

# 添加国庆后工作日字段
def addAfterNational(df, dayLen):
    df['is_after_national%d'%dayLen] = 0
    for y in df.year.value_counts().index:
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-10-01'%y):('%d-10-15'%y)][holidayDf.holiday==0].index[:dayLen]
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

# 添加销售量数据
def addSaleFea(df):
    saleDf = importDf('../extra_data/yancheng_sale.csv', sep=',')
    saleDf = pd.pivot_table(saleDf, index=['sale_date'], values='sale_quantity', aggfunc=np.sum)
    saleDf.loc[201711, 'sale_quantity'] = saleDf.loc[201611, 'sale_quantity'] * 1.025
    saleDf.columns = ['sale_month']
    saleDf.reset_index(inplace=True)
    saleDf.index = pd.to_datetime(saleDf['sale_date'].astype(str), format='%Y%m')
    saleDf['sale_last_month'] = saleDf['sale_month'].shift(1, freq='MS')
    saleDf['sale_last_year'] = saleDf['sale_month'].shift(12, freq='MS')
    saleDf['sale_yoy'] = (saleDf['sale_month'].values - saleDf['sale_last_year'].values) / saleDf['sale_last_year'].values
    saleDf['sale_last_yoy'] = saleDf['sale_yoy'].shift(1, freq='MS')
    saleDf.loc['2013-01-01','sale_last_yoy'] = 0.2416
    df = pd.merge(df, saleDf, how='left', left_on='year_month', right_on='sale_date')
    return df

# 添加历史上牌统计数据
def addHistoryFea(df):
    brandDf = pd.pivot_table(df, index=['brand', 'holiday'], values='cnt', aggfunc=[np.mean, np.median])
    brandDf.columns = ['cnt_mean', 'cnt_median']
    brandDf.reset_index(inplace=True)
    df = df.merge(brandDf, how='left', on=['brand', 'holiday'])
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
    df = addAfterNewyear(df, 3)
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
    df = addHistoryFea(df)
    df = addOneHot(df, ['day_of_week','year','month','holiday','brand'])
    return df

# 训练模型
def trainModel(X, y, showCoef=True, showAlpha=True):
    clf = linear_model.RidgeCV(alphas=[0.02*x for x in range(1,500)], scoring='neg_mean_squared_error')
    clf.fit(X, y)
    if showCoef:
        print('Coefficients:', clf.coef_)
    if showAlpha:
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
    deltaSeries.plot(style='b-')
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
    df = importDf('../data/fusai_train_20180227.txt')

    # 特征提取
    startTime = datetime.now()
    df = feaFactory(df)
    scaleCols = ['year','sale_last_month','sale_month']
    df,scaler = scalerFea(df, scaleCols)
    df.loc[(df.is_sunday_holiday==1)&(df.cnt>150)] = None
    df = df.dropna()
    print("feature time: ", datetime.now() - startTime)
    print("训练集：\n",df.tail())
    fea = [
        'is_sunday_holiday',
        'sale_month',#'sale_last_month',
        'is_after_newyear3','is_after_newyear5',#'after_new_year_weight',
        'is_before_spring_fest1','is_before_spring_fest5','is_after_spring_fest3','is_before_spring_fest9',
        'is_after_worker1',
        'is_after_national1'
        ]
    fea.extend(['month_%d'%x for x in [1,2,3,4,5,6,7,8,9,10,11,12]])
    fea.extend(['day_of_week_%d'%x for x in [1,2,3,4,5,6]])
    fea.extend(['holiday_%d'%x for x in range(0,3)])
    # 填补缺失字段
    for x in [x for x in fea if x not in df.columns]:
        df[x] = 0
    print("模型输入：\n",df[fea].info())

    # 用滑动窗口检验模型
    costDf = {b:pd.DataFrame(index=fea+['cost']) for b in range(1,11)}
    dtCostDf = pd.DataFrame(columns=['cost'])
    clfs = {}
    for dt in pd.date_range(start='2015-01-01', end='2015-05-01', freq='MS'):
        splitDate = dt
        trainDf = df[(df.guess_date < splitDate)]
        testDf = df[(df.guess_date >= splitDate) & (df.guess_date < splitDate+timedelta(days=365))]
        trainDf['predict'] = np.nan
        testDf['predict'] = np.nan
        for b in df.brand.value_counts().index:
            clf = trainModel(trainDf[trainDf.brand==b][fea].values, trainDf[trainDf.brand==b]['cnt'].values, showCoef=False)
            clfs[b] = clf
            testDf.loc[testDf.brand==b,'predict'] = clf.predict(testDf[testDf.brand==b][fea].values)
            testDf.loc[:,'predict'] = testDf['predict'].map(lambda x: 0 if x<0 else x)   #修正负数值
            testDf.loc[testDf.is_sunday_holiday==1,'predict'] = testDf.loc[testDf.is_sunday_holiday==1,'predict'].map(lambda x: x-100 if x>100 else x)   #修正周日休息日的数据
            cost = metrics.mean_squared_error(testDf[testDf.brand==b]['cnt'].values, testDf[testDf.brand==b]['predict'].values) 
            costDf[b][dt] = list(clf.coef_)+[cost]
        cost = metrics.mean_squared_error(testDf['cnt'].values, testDf['predict'].values)
        dtCostDf.loc[dt, 'cost'] = cost
    for b,cdf in costDf.items():
        cdf['mean'] = cdf.mean(axis=1)
        cdf['std'] = cdf.std(axis=1)
    print(costDf)
    print(dtCostDf)
    # 绘制误差曲线
    for b in range(1,11):
        deltaSeries = countDeltaY(testDf[testDf.brand==b].set_index(['guess_date'])['predict'], testDf[testDf.brand==b].set_index(['guess_date'])['cnt'], show=False, title='brand%d'%b, subplot=(5,2,b))
    plt.show()

    # 导出完整训练集预测记录
    df['predict'] = df['delta'] = np.nan
    for b in range(1,11):
        df.loc[df.brand==b,'predict'], _ = getOof(clfs[b], df.loc[df.brand==b,fea].values, df.loc[df.brand==b,'cnt'].values, testDf.loc[df.brand==b,fea].values, nFold=10)
    df['delta'] = df['predict'] - df['cnt']
    cost = metrics.mean_squared_error(df['cnt'].values, df['predict'].values) 
    print('cv cost:', cost)
    exportResult(df[['date','guess_date','brand','cnt','predict','delta','day_of_week','holiday','sale_month']], "train_predict.csv", header=True)
    exit()


    # 正式模型
    modelName = "linear2A"
    clfs = {}
    for b in range(1,11):
        clf = trainModel(df[df.brand==b][fea].values, df[df.brand==b]['cnt'].values)
        joblib.dump(clf, './%s-b%d.pkl' % (modelName,b), compress=3)
        clfs[b] = clf

    # 预测集准备
    startTime = datetime.now()
    predictDf = importDf('../data/fusai_test_A_20180227.txt')
    predictDf = feaFactory(predictDf, startWeek=df.loc[df.index[-1], 'week'])
    # 填补缺失字段
    for x in [x for x in fea if x not in predictDf.columns]:
        predictDf[x] = 0
    predictDf[scaleCols] = scaler.transform(predictDf[scaleCols].values)
    print("预测集：\n",predictDf.head())
    print(predictDf[fea].info())

    # 开始预测
    predictDf['predict'] = np.nan
    for b in range(1,11):
        predictDf.loc[predictDf.brand==b,'predict'] = clfs[b].predict(predictDf[predictDf.brand==b][fea].values)
    print("预测结果：\n",predictDf[['date','brand','predict']].head())
    exportResult(predictDf[['date','guess_date','brand','predict','day_of_week','holiday','sale_month']], "%s_predict.csv" % modelName, header=True)
    predictDf['predict'] = predictDf['predict'].map(lambda x: 0 if x<0 else x)   #修正负数值
    predictDf.loc[predictDf.is_sunday_holiday==1,'predict'] = predictDf.loc[predictDf.is_sunday_holiday==1,'predict'].map(lambda x: x-100 if x>100 else x)   #修正周日休息日的数据
    exportResult(predictDf[['date','brand','predict']], "%s.txt" % modelName, sep='\t')

    # 生成模型融合数据集
    df['predict'] = np.nan
    predictDf['predict'] = np.nan
    for b in range(1,11):
        df.loc[df.brand==b,'predict'], predictDf.loc[predictDf.brand==b,'predict'] = getOof(clfs[b], df.loc[df.brand==b,fea].values, df.loc[df.brand==b,'cnt'].values, predictDf.loc[predictDf.brand==b,fea].values, nFold=10)
    predictDf['predict'] = predictDf['predict'].map(lambda x: 0 if x<0 else x)   #修正负数值
    exportResult(df[['date','brand','predict']], "%s_oof_train.csv" % modelName, header=True)
    exportResult(predictDf[['date','brand','predict']], "%s_oof_testA.csv" % modelName, header=True)

