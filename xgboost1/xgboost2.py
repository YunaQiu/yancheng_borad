#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： xgboostCV
模型参数：'objective': 'reg:linear',
        'eval_metric':'rmse',
        'silent': True,
        'eta': 0.1,
        'max_depth': 6,
        'gamma': 0,
        'min_child_weight': 7,
        'subsample':1,
        'colsample_bytree': 0.9
        'alpha':1600,
        'num_boost_round': 1500
        'early_stopping_rounds': 2
        'nfold': 10
前期处理：训练前经过数据清洗：舍弃部分品牌前期的异常数据
        参数中max_depth/gamma/min_child_weight/subsample/colsample_bytree/alpha采用10折的网格搜索确定
        训练模型时采用10折cv确定树的棵树后，再用全部数据训练一次
后期处理：假日类型2的预测结果用各自品牌的假日类型2的历史中位数代替
        假日类型0且预测结果小于30的预测值，加多100的预测量
特征： 星期1~7的onehot标记
      月份1~12的onehot标记
      节假日类型0/1/2的onehot标记
      是否周日休息日
      元旦前5个工作日权重，元旦后5个工作日权重
      春节前9个工作日权重，春节后5个工作日权重
      五一后1个工作日权重
      国庆前5个工作日权重，国庆后3个工作日权重
结果： 线下（51789）

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
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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
def cleanDf(df, beforeDate='2018-01-01'):
    tempDf = df[df.guess_date<beforeDate].copy()
    # tempDf['unusual'] = tempDf.apply(lambda x: x.cnt > tempDf[(tempDf.brand==x.brand)&(abs(tempDf.date-x.date)<7)&(tempDf.date-x.date!=0)]['cnt'].max() * 1.8, axis=1)
    # tempDf.loc[tempDf.unusual,'unusual'] = tempDf[tempDf.unusual].apply(lambda x: tempDf[(tempDf.unusual)&(tempDf.date==x.date)].brand.count() < 3, axis=1)

    cleanIndex = set()
    # cleanIndex |= set(tempDf[tempDf.unusual].index)
    cleanIndex |= set(tempDf[(tempDf.holiday==1)&(tempDf.cnt>400)].index)
    cleanIndex |= set(tempDf[(tempDf.holiday==2)&(tempDf.cnt>70)].index)
    cleanIndex |= set(tempDf[(tempDf.is_sunday_holiday==1)&(tempDf.cnt>150)].index)
    cleanIndex |= set(tempDf[(tempDf.brand==8)&(tempDf.guess_date<'2015-09-01')].index)
    cleanIndex |= set(tempDf[(tempDf.brand==5)&(tempDf.guess_date<'2015-11-01')].index)
    cleanIndex |= set(tempDf[(tempDf.brand==2)&(tempDf.guess_date<'2014-07-01')].index)
    cleanIndex |= set(tempDf[(tempDf.brand==9)&(tempDf.guess_date<'2014-03-01')].index)
    cleanIndex |= set(tempDf[(tempDf.brand==6)&(tempDf.guess_date<'2014-03-01')].index)
    # cleanIndex |= set(tempDf[(tempDf.brand==7)&(tempDf.guess_date<'2015-03-01')&(tempDf.holiday>0)].index)
    # cleanIndex |= set(tempDf[(tempDf.brand==10)&(tempDf.guess_date<'2013-07-01')].index)
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
        2013:'2013-09-17',
        2014:'2014-09-17',
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
    saleDf['sale_month_predict'] = saleDf['sale_last_year'] * (1 + saleDf['sale_last_yoy'])
    df = pd.merge(df, saleDf, how='left', left_on='year_month', right_on='sale_date')
    return df

# 添加历史上牌统计数据
def addHistoryFea(df, cntDf=None):
    if not(isinstance(cntDf, pd.DataFrame)):
        cntDf = df.copy()
    brandDf = cntDf.sort_values(by=['brand','date','holiday']).set_index(['brand','date','holiday'])[['cnt']]
    df['cnt_his'] = df[['brand','date','holiday']].apply(lambda x: list(brandDf.loc[pd.IndexSlice[x.brand,:x.date,x.holiday],'cnt'].values), axis=1).values
    df['cnt_holiday_mean'] = df['cnt_his'].map(lambda x: np.mean(x))
    df['cnt_holiday_median'] = df['cnt_his'].map(lambda x: np.median(x))
    # df['cnt_holiday_max'] = df['cnt_his'].map(lambda x: np.max(x))
    # df['cnt_holiday_min'] = df['cnt_his'].map(lambda x: np.min(x))

    brandDf = pd.pivot_table(cntDf, index='brand', values='date', aggfunc=np.min)
    brandDf.columns = ['start_date']
    df = df.merge(brandDf, how='left', left_on='brand', right_index=True)

    # monthDf = cntDf.sort_values(by=['brand','date','month']).set_index(['brand','date','month'])[['cnt']]
    # df['cnt_his'] = df[['brand','date','month']].apply(lambda x: list(monthDf.loc[pd.IndexSlice[x.brand,:x.date,x.month],'cnt'].values), axis=1).values
    # df['cnt_month_mean'] = df['cnt_his'].map(lambda x: np.mean(x))
    # df['cnt_month_median'] = df['cnt_his'].map(lambda x: np.median(x))
    # df['cnt_month_max'] = df['cnt_his'].map(lambda x: np.max(x))
    # df['cnt_month_min'] = df['cnt_his'].map(lambda x: np.min(x))
    df.drop(['cnt_his'], axis=1, inplace=True)
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
    df = addAfterCarShow(df, 20)
    # df = addSaleFea(df)
    if startWeek==0:
        cntDf = cleanDf(df.copy())
    df = addHistoryFea(df, cntDf=cntDf)
    df = addOneHot(df, ['day_of_week','year','month','holiday','brand'])
    return df

class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            'objective': 'reg:linear',
            'eval_metric':'rmse',
            'silent': True,
            'eta': 0.1,
            'max_depth': 6,
            'gamma': 0,
            'subsample':1,
            'colsample_bytree': 0.9,
            'min_child_weight': 7,
            'alpha':1600
        }
        for k,v in params.items():
            self.params[k] = v
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=1000, early_stopping_rounds=3):
        if train_size==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feaNames)
        dval = xgb.DMatrix(X_test, label=y_test, feature_names=self.feaNames)
        watchlist = [(dtrain,'train'),(dval,'val')]
        clf = xgb.train(
            self.params, dtrain, 
            num_boost_round = num_boost_round, 
            evals = watchlist, 
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        self.clf = clf

    def trainCV(self, X, y, nFold=10, verbose=True, num_boost_round=1500, early_stopping_rounds=2):
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        cvResult = xgb.cv(
            self.params, dtrain, 
            num_boost_round = num_boost_round, 
            nfold = nFold,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        clf = xgb.train(
            self.params, dtrain, 
            num_boost_round = cvResult.shape[0], 
        )
        self.clf = clf

    def gridSearch(self, X, y, nFold=10, verbose=1, num_boost_round=70, early_stopping_rounds=30):
        paramsGrids = {
            # 'n_estimators': [70+5*i for i in range(0,10)]
            # 'gamma': [100000+100000*i for i in range(0,10)],
            # 'max_depth': list(range(4,12)),
            # 'min_child_weight': list(range(5,15))
            # 'subsample': [1,0.95,0.9,0.85,0.8]
            # 'colsample_bytree': [1,0.95,0.9,0.85,0.8]
            'reg_alpha':[1000+100*i for i in range(0,20)]
        }
        gsearch = GridSearchCV(
            estimator = xgb.XGBRegressor(
                max_depth = self.params['max_depth'], 
                gamma = self.params['gamma'],
                learning_rate = self.params['eta'],
                min_child_weight =  self.params['min_child_weight'],
                subsample = self.params['subsample'],
                colsample_bytree = self.params['colsample_bytree'],
                silent = self.params['silent'],
                # reg_alpha = self.params['alpha'],
                n_estimators = num_boost_round
            ),
            param_grid = paramsGrids,
            scoring = 'neg_mean_squared_error',
            cv = nFold,
            verbose = verbose,
            n_jobs = 3
        )
        gsearch.fit(X, y)
        print(pd.DataFrame(gsearch.cv_results_))
        print(gsearch.best_params_)
        exit()

    def predict(self, X):
        return self.clf.predict(xgb.DMatrix(X, feature_names=self.feaNames))

    def getFeaScore(self, show=False):
        fscore = self.clf.get_score()
        feaNames = fscore.keys()
        scoreDf = pd.DataFrame(index=feaNames, columns=['importance'])
        for k,v in fscore.items():
            scoreDf.loc[k, 'importance'] = v
        if show:
            print(scoreDf.sort_index(by=['importance'], ascending=False))
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
        clf.trainCV(kfTrainX, kfTrainY, verbose=False)
        oofTrain[testIdx] = clf.predict(kfTestX)
    clf.trainCV(trainX,trainY, verbose=False)
    oofTest = clf.predict(testX)
    return oofTrain, oofTest


if __name__ == '__main__':
    # 导入数据
    df = importDf('../data/fusai_train_20180227.txt')
    dfA = importDf('../data/fusai_test_A_20180227.txt')
    answerDfA = importDf('../data/fusai_answer_a_20180307.txt', header=None, colNames=['date','brand','cnt'])
    dfA.drop(0,inplace=True)
    dfA = dfA.merge(answerDfA, how='left', on=['date','brand'])
    df = pd.concat([df, dfA], ignore_index=True)  

    # 特征提取
    startTime = datetime.now()
    df = feaFactory(df)
    print("feature time: ", datetime.now() - startTime)
    print("训练集：\n",df.tail())
    fea = [
        # 'cnt_holiday_median','cnt_holiday_mean',#'cnt_holiday_max','cnt_holiday_min',
        # 'cnt_month_median','cnt_month_mean',#'cnt_month_max','cnt_month_min',
        'is_sunday_holiday',
        # 'start_date',
        # 'sale_last_month','sale_last_yoy',#'sale_month_predict','sale_last_year'
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
    for dt in pd.date_range(start='2015-06-01', end='2015-10-01', freq='MS'):
        # 准备训练/测试集
        splitDate = dt
        trainDf = df[(df.guess_date < splitDate)]
        trainDf = cleanDf(trainDf, dt)
        testDf = df[(df.guess_date >= splitDate) & (df.guess_date < splitDate+timedelta(days=365))]
        # 训练并统计结果
        # xgbModel.gridSearch(trainDf[fea].values, trainDf['cnt'].values, nFold=10)
        xgbModel.trainCV(trainDf[fea].values, trainDf['cnt'].values, nFold=10)
        testDf.loc[:,'predict'] = xgbModel.predict(testDf[fea].values)
        testDf.loc[(testDf.holiday>1), 'predict'] = testDf.loc[(testDf.holiday>1), 'cnt_holiday_median']
        testDf['predict'] = testDf.apply(lambda x: x.predict+100 if (x.holiday==0)&(x.predict<30) else x.predict, axis=1)
        testDf.loc[(testDf.predict<0),'predict'] = 5

        scoreDf = xgbModel.getFeaScore()
        scoreDf.columns = [dt.strftime('%Y-%m')]
        costDf = costDf.merge(scoreDf, how='left', left_index=True, right_index=True)
        cost = metrics.mean_squared_error(testDf['cnt'].values, testDf['predict'].values)
        costDf.loc['cost',dt.strftime('%Y-%m')] = cost
    print(costDf)
    # 绘制误差曲线
    for b in range(1,11):
        deltaSeries = countDeltaY(testDf[testDf.brand==b].set_index(['guess_date'])['predict'], testDf[testDf.brand==b].set_index(['guess_date'])['cnt'], show=False, title='brand%d'%b, subplot=(5,2,b))
        print('brand%d cost:'%b,metrics.mean_squared_error(testDf[testDf.brand==b]['cnt'].values, testDf[testDf.brand==b]['predict'].values))
    plt.show()

    # 导出完整训练集预测记录
    startTime = datetime.now()
    df['predict'] = df['delta'] = np.nan
    df.loc[:,'predict'], predict = getOof(xgbModel, df.loc[:,fea].values, df.loc[:,'cnt'].values, testDf.loc[:,fea].values, nFold=10)
    df['delta'] = df['predict'] - df['cnt']
    cost = metrics.mean_squared_error(df['cnt'].values, df['predict'].values) 
    print('cv cost:', cost)
    deltaSeries = countDeltaY(df.set_index(['guess_date'])['predict'], df.set_index(['guess_date'])['cnt'], show=False)
    exportResult(df[['date','guess_date','brand','cnt','predict','delta','day_of_week','holiday']], "train_predict.csv", header=True)
    print("export trainSet time: ", datetime.now() - startTime)
    # exit()


    # 正式模型
    modelName = "xgboost2B"
    xgbModel.trainCV(df[fea].values, df['cnt'].values)
    xgbModel.getFeaScore(show=True)
    xgbModel.clf.save_model('%s.model'%modelName)

    # 预测集准备
    startTime = datetime.now()
    predictDf = importDf('../data/fusai_test_B_20180227.txt')
    predictDf = feaFactory(predictDf, startWeek=df.loc[df.index[-1], 'week'], cntDf=df)
    # 填补缺失字段
    for x in [x for x in fea if x not in predictDf.columns]:
        predictDf[x] = 0
    print("预测集：\n",predictDf.head())
    print(predictDf[fea].info())

    # 开始预测
    predictDf.loc[:,'predict'] = xgbModel.predict(predictDf[fea].values)
    print("预测结果：\n",predictDf[['date','brand','predict']].head())
    exportResult(predictDf[['date','guess_date','brand','predict','day_of_week','holiday']], "%s_predict.csv" % modelName, header=True)
    print(predictDf[predictDf.predict<0][['date','guess_date','brand','predict','day_of_week','holiday']])
    predictDf.loc[(predictDf.holiday>1), 'predict'] = predictDf.loc[(predictDf.holiday>1), 'cnt_holiday_median']
    predictDf['predict'] = predictDf.apply(lambda x: x.predict+100 if (x.holiday==0)&(x.predict<30) else x.predict, axis=1)
    predictDf.loc[(predictDf.predict<0),'predict'] = 5
    exportResult(predictDf[['date','brand','predict']], "%s.txt" % modelName, sep='\t')
    # exit()

    # 生成模型融合数据集
    df.loc[:,'predict'], predictDf.loc[:,'predict'] = getOof(xgbModel, df[fea].values, df['cnt'].values, predictDf[fea].values, nFold=10)
    predictDf['predict'] = predictDf['predict'].map(lambda x: 0 if x<0 else x)   #修正负数值
    exportResult(df[['date','brand','predict']], "%s_oof_train.csv" % modelName, header=True)
    exportResult(predictDf[['date','brand','predict']], "%s_oof_testA.csv" % modelName, header=True)

