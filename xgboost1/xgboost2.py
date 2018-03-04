#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： xgboost
模型参数：'objective': 'reg:linear',
        'eval_metric':'rmse',
        'silent': True,
        'eta': 0.05,
        'max_depth': 7,
        'gamma': 300,
        'subsample':1,
        'colsample_bytree': 0.9
        'num_boost_round': 1000
        'early_stopping_rounds': 3
特殊处理：训练前经过数据清洗：去除前后7天内异常拔高的数据（除非当天超过3个品牌有拔高情况）
        训练时的模型验证集来自训练集中随机抽取的10%的数据
特征： 星期1~7的onehot标记
      月份1~12的onehot标记
      节假日类型0/1/2的onehot标记
      历史同假日类型的上牌量中位数/平均数
      历史同月份的上排量中位数/平均数
      是否周日休息日
      本月销售量/上个月销售量（取自零售量比赛数据，其中17年11月数据根据网上的16年同期增长比估算）
      元旦前5个工作日权重，元旦后5个工作日权重
      春节前9个工作日权重，春节后5个工作日权重
      五一后1个工作日权重
      国庆前5个工作日权重，国庆后3个工作日权重
结果： A榜（39036）

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

# 异常数据清除
def cleanDf(df):
    tempDf = df.copy()
    tempDf['unusual'] = tempDf.apply(lambda x: x.cnt > tempDf[(tempDf.brand==x.brand)&(abs(tempDf.date-x.date)<7)&(tempDf.date-x.date!=0)]['cnt'].max() * 1.8, axis=1)
    tempDf.loc[tempDf.unusual,'unusual'] = tempDf[tempDf.unusual].apply(lambda x: tempDf[(tempDf.unusual)&(tempDf.date==x.date)].brand.count() < 4, axis=1)

    cleanIndex = set()
    cleanIndex |= set(tempDf[tempDf.unusual].index)
    cleanIndex |= set(df[(df.holiday==1)&(df.cnt>400)].index)
    cleanIndex |= set(df[(df.holiday==2)&(df.cnt>70)].index)
    cleanIndex |= set(df[(df.is_sunday_holiday==1)&(df.cnt>150)].index)
    df.drop(cleanIndex, inplace=True)
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
    df['is_after_worker%d'%dayLen] = df['after_worker_weight'] = 0
    for y in df.year.value_counts().index:
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-05-01'%y):('%d-05-10'%y)][holidayDf.holiday==0].index[:dayLen]
        df.loc[df.guess_date.isin(dateList), 'is_after_worker%d'%dayLen] = 1

        weightSeries = pd.Series(list(range(dayLen,0,-1)), index=dateList)
        df.loc[df.guess_date.isin(dateList), 'after_worker_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
    return df

# 添加国庆前工作日字段
def addBeforeNational(df, dayLen):
    df['is_before_national%d'%dayLen] = df['before_national_weight'] = 0
    for y in df.year.value_counts().index:
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-09-01'%y):('%d-10-01'%y)][holidayDf.holiday==0].index[-dayLen:]
        df.loc[df.guess_date.isin(dateList), 'is_before_national%d'%dayLen] = 1

        weightSeries = pd.Series(list(range(1,dayLen+1)), index=dateList)
        df.loc[df.guess_date.isin(dateList), 'before_national_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
    return df

# 添加国庆后工作日字段
def addAfterNational(df, dayLen):
    df['is_after_national%d'%dayLen] = df['after_national_weight'] = 0
    for y in df.year.value_counts().index:
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-10-01'%y):('%d-10-15'%y)][holidayDf.holiday==0].index[:dayLen]
        df.loc[df.guess_date.isin(dateList), 'is_after_national%d'%dayLen] = 1

        weightSeries = pd.Series(list(range(dayLen,0,-1)), index=dateList)
        df.loc[df.guess_date.isin(dateList), 'after_national_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
    return df

# 添加车展节前工作日字段
def addBeforeCarShow(df, dayLen):
    df['before_carshow_weight'] = 0
    showDate = {
        2015:'2015-09-18',
        2016:'2016-09-14',
        2017:'2017-09-15'}
    for y in df.year.value_counts().index:
        if y not in showDate.keys():
            continue
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[('%d-08-15'%y):showDate[y]][holidayDf.holiday==0].index[-dayLen:]

        weightSeries = pd.Series(list(range(1,dayLen+1)), index=dateList)
        df.loc[df.guess_date.isin(dateList), 'before_carshow_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
    return df

# 添加车展节后工作日字段
def addAfterCarShow(df, dayLen):
    df['after_carshow_weight'] = 0
    showDate = {
        2015:'2015-09-18',
        2016:'2016-09-14',
        2017:'2017-09-15'}
    for y in df.year.value_counts().index:
        if y not in showDate.keys():
            continue
        holidayDf = getHolidayDf()
        dateList = holidayDf.loc[showDate[y]:('%d-10-30'%y)][holidayDf.holiday==0].index[:dayLen]

        weightSeries = pd.Series(list(range(dayLen,0,-1)), index=dateList)
        df.loc[df.guess_date.isin(dateList), 'after_carshow_weight'] = df.loc[df.guess_date.isin(dateList), 'guess_date'].map(lambda x: weightSeries[x])
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
def addHistoryFea(df, cntDf=None):
    if not(isinstance(cntDf, pd.DataFrame)):
        cntDf = df.copy()
    brandDf = pd.pivot_table(cntDf, index=['brand', 'holiday'], values='cnt', aggfunc=[np.mean, np.median])
    brandDf.columns = ['cnt_holiday_mean', 'cnt_holiday_median']
    brandDf.reset_index(inplace=True)
    df = df.merge(brandDf, how='left', on=['brand', 'holiday'])

    monthDf = pd.pivot_table(cntDf, index=['month','brand'], values='cnt', aggfunc=[np.mean, np.median])
    monthDf.columns = ['cnt_month_mean', 'cnt_month_median']
    monthDf.reset_index(inplace=True)
    df = df.merge(monthDf, how='left', on=['brand', 'month'])
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
def feaFactory(df, startWeek=0, cntDf=None):
    df = tickWeek(df, startWeek)
    df = addGuessDate(df,'2012-12-30')
    df = addHoliday(df)
    df = addSundayHoliday(df)
    df = addBeforeNewyear(df, 5)
    df = addAfterNewyear(df, 5)
    df = addBeforeSpringFest(df, 9)
    df = addAfterSpringFest(df, 5)
    df = addAfterWorker(df, 3)
    df = addBeforeNational(df, 5)
    df = addAfterNational(df, 3)
    df = addBeforeCarShow(df, 5)
    df = addAfterCarShow(df, 15)
    df = addSaleFea(df)
    df = addHistoryFea(df, cntDf=cntDf)
    df = addOneHot(df, ['day_of_week','year','month','holiday','brand'])
    return df

class XgbModel:
    def __init__(self, type='xgb', feaNames=None):
        self.type = type
        self.feaNames = feaNames
        self.params = {
            'objective': 'reg:linear',
            'eval_metric':'rmse',
            'silent': True,
            'eta': 0.05,
            'max_depth': 7,
            'gamma': 300,
            'subsample':1,
            'colsample_bytree': 0.9
        }
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=1000, early_stopping_rounds=3):
        if train_size==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        if self.type=='xgb':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_test, label=y_test)
            watchlist = [(dtrain,'train'),(dval,'val')]
            clf = xgb.train(
                self.params, dtrain, 
                num_boost_round = num_boost_round, 
                evals = watchlist, 
                early_stopping_rounds = early_stopping_rounds,
                verbose_eval=verbose
            )
            self.clf = clf
        elif self.type=='skl':
            clf = xgb.XGBRegressor(
                max_depth = self.params['max_depth'], 
                gamma = self.params['gamma'],
                learning_rate = self.params['eta'],
                subsample = self.params['subsample'],
                colsample_bytree = self.params['colsample_bytree'],
                silent = self.params['silent'],
                n_estimators = num_boost_round,
                verbose = (True if isinstance(verbose,int) else verbose)
            )
            clf = clf.fit(
                X_train, y_train, 
                eval_metric = self.params['eval_metric'], 
                eval_set = [(X_test, y_test)],
                early_stopping_rounds = early_stopping_rounds
            )
            self.clf = clf

    def predict(self, X):
        if self.type=='xgb':
            return self.clf.predict(xgb.DMatrix(X))
        elif self.type=='skl':
            return self.clf.predict(X)

    def getFeaScore(self, show=False):
        feaNames = self.feaNames
        if feaNames==None:
            feaNames = fscore.keys()
        scoreDf = pd.DataFrame(feaNames, columns=['fea'])
        scoreDf['importance'] = np.nan
        if self.type=='xgb':
            fscore = self.clf.get_score()
        elif self.type=='skl':
            fscore = self.clf.get_booster().get_score()
        for k,v in fscore.items():
            scoreDf.loc[int(k[1:]), 'importance'] = v
        scoreDf.set_index('fea', inplace=True)
        if show:
            print(scoreDf.dropna().sort_index(by=['importance'], ascending=False))
        return scoreDf

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
        clf.train(kfTrainX, kfTrainY, verbose=False)
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
    print("feature time: ", datetime.now() - startTime)
    print("训练集：\n",df.tail())
    fea = [
        'cnt_holiday_median','cnt_holiday_mean',
        'cnt_month_mean', 'cnt_month_median',
        'is_sunday_holiday',
        'sale_month','sale_last_month',
        'before_new_year_weight','after_new_year_weight',
        'before_spring_fest_weight','after_spring_fest_weight',
        'after_worker_weight',
        'before_national_weight','after_national_weight',
        # 'before_carshow_weight','after_carshow_weight'
        ]
    fea.extend(['month_%d'%x for x in [1,2,3,4,5,6,7,8,9,10,11,12]])
    fea.extend(['day_of_week_%d'%x for x in [1,2,3,4,5,6,7]])
    fea.extend(['holiday_%d'%x for x in range(0,3)])
    fea.extend(['brand_%d'%x for x in range(1,11)])
    # 填补缺失字段
    for x in [x for x in fea if x not in df.columns]:
        df[x] = 0
    print("模型输入：\n",df[fea].info())
    # print("训练特征:",fea)

    # 用滑动窗口检验模型
    costDf = pd.DataFrame(index=fea+['cost'])
    xgbModel = XgbModel(feaNames=fea)
    for dt in pd.date_range(start='2015-01-01', end='2015-05-01', freq='MS'):
        # 准备训练/测试集
        splitDate = dt
        trainDf = df[(df.guess_date < splitDate)]
        trainDf = cleanDf(trainDf)
        testDf = df[(df.guess_date >= splitDate) & (df.guess_date < splitDate+timedelta(days=365))]
        trainDf.loc[:,'predict'] = np.nan
        testDf.loc[:,'predict'] = np.nan
        # 训练并统计结果
        xgbModel.train(trainDf[fea].values, trainDf['cnt'].values, num_boost_round=1000)
        testDf.loc[:,'predict'] = xgbModel.predict(testDf[fea].values)
        scoreDf = xgbModel.getFeaScore()
        scoreDf.columns = [dt.strftime('%Y-%m')]
        costDf = costDf.merge(scoreDf, how='left', left_index=True, right_index=True)
        cost = metrics.mean_squared_error(testDf['cnt'].values, testDf['predict'].values)
        costDf.loc['cost',dt.strftime('%Y-%m')] = cost
    print(costDf)
    # 绘制误差曲线
    for b in range(1,11):
        deltaSeries = countDeltaY(testDf[testDf.brand==b].set_index(['guess_date'])['predict'], testDf[testDf.brand==b].set_index(['guess_date'])['cnt'], show=False, title='brand%d'%b, subplot=(5,2,b))
    plt.show()

    # 导出完整训练集预测记录
    df['predict'] = df['delta'] = np.nan
    df.loc[:,'predict'], predict = getOof(xgbModel, df.loc[:,fea].values, df.loc[:,'cnt'].values, testDf.loc[:,fea].values, nFold=10)
    df['delta'] = df['predict'] - df['cnt']
    cost = metrics.mean_squared_error(df['cnt'].values, df['predict'].values) 
    print('cv cost:', cost)
    deltaSeries = countDeltaY(df.set_index(['guess_date'])['predict'], df.set_index(['guess_date'])['cnt'], show=False)
    exportResult(df[['date','guess_date','brand','cnt','predict','delta','day_of_week','holiday','sale_month']], "train_predict.csv", header=True)
    exit()


    # 正式模型
    modelName = "xgboost2A_merge"
    df = cleanDf(df)
    xgbModel.train(df[fea].values, df['cnt'].values)
    xgbModel.getFeaScore(show=True)
    xgbModel.clf.save_model('%s.model'%modelName)
    # exit()

    # 预测集准备
    startTime = datetime.now()
    predictDf = importDf('../data/fusai_test_A_20180227.txt')
    predictDf = feaFactory(predictDf, startWeek=df.loc[df.index[-1], 'week'], cntDf=df)
    # 填补缺失字段
    for x in [x for x in fea if x not in predictDf.columns]:
        predictDf[x] = 0
    print("预测集：\n",predictDf.head())
    print(predictDf[fea].info())

    # 开始预测
    predictDf.loc[:,'predict'] = xgbModel.predict(predictDf[fea].values)
    print("预测结果：\n",predictDf[['date','brand','predict']].head())
    exportResult(predictDf[['date','guess_date','brand','predict','day_of_week','holiday','sale_month']], "%s_predict.csv" % modelName, header=True)
    print(predictDf[predictDf.predict<0])
    predictDf['predict'] = predictDf['predict'].map(lambda x: 0 if x<0 else x)   #修正负数值
    exportResult(predictDf[['date','brand','predict']], "%s.txt" % modelName, sep='\t')
    exit()

    # 生成模型融合数据集
    df.loc[:,'predict'], predictDf.loc[:,'predict'] = getOof(clfs[b], df[fea].values, df['cnt'].values, predictDf[fea].values, nFold=10)
    predictDf['predict'] = predictDf['predict'].map(lambda x: 0 if x<0 else x)   #修正负数值
    exportResult(df[['date','brand','predict']], "%s_oof_train.csv" % modelName, header=True)
    exportResult(predictDf[['date','brand','predict']], "%s_oof_testA.csv" % modelName, header=True)

