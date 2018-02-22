#!/usr/bin/env python
# -*-coding:utf-8-*-

import urllib, urllib.parse, urllib.request
from bs4 import BeautifulSoup
import re
import csv
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from datetime import *

# 构建请求url
def getRequestUrl(cityCode, dateCode, domain="http://www.tianqihoubao.com/lishi", formatter="%s/%s/month/%s.html"):
    url = formatter % (domain, cityCode, dateCode)
    return url

# 通过http获取的html源代码对象
def getHTMLSoup(url):
    html = urllib.request.urlopen(url).read()
    return BeautifulSoup(html,'lxml')

# 替换字符串中多余的换行符制表符及空格
def stripString(string, replaceSpace='',replaceEnter=''):
    if not(replaceSpace==None):
        pattern = re.compile('[ \t]+')
        string = pattern.sub(replaceSpace, string)
    if not(replaceEnter==None):
        pattern = re.compile('[\r\n]+')
        string = pattern.sub(replaceEnter, string)
    return string

# 将table转成dataframe
def htmlTableToDf(html, header=True):
    tbody = rowList = html.select('tr')
    colName = list()
    content = list()
    # 提取表头信息
    if header:
        thead = rowList[0]
        tbody = rowList[1:]
        if len(thead.select('th'))>0:
            for td in thead.select('th'):
                ceil = ''
                for string in td.stripped_strings :
                    ceil += stripString(string)
                colName.append(ceil)
        else:
            for td in thead.select('td'):
                ceil = ''
                for string in td.stripped_strings :
                    ceil += stripString(string)
                colName.append(ceil)
    # 提取正文数据
    for row in tbody:
        rowContent = list()
        for td in row.select('td'):
            ceil = ''
            for string in td.stripped_strings :
                    ceil += stripString(string, replaceSpace=' ')
            rowContent.append(ceil)
        content.append(rowContent)
    # 生成dataframe
    if len(colName)>0:
        df = pd.DataFrame(content, columns=colName)
    else:
        df = pd.DataFrame(content)
    return df

# 分割白天天气和夜间天气
def addDayNightStatus(df):
    df['status_day'] = df['status'].map(lambda x: x[:x.find('/')].strip(' '))
    df['status_night'] = df['status'].map(lambda x: x[(x.find('/')+1):].strip(' '))
    return df

# 分割白天气温和夜间气温
def addDayNightTemp(df):
    df['temp_day'] = df['temperature'].map(lambda x: x[:x.find('/')].strip('[ ℃]'))
    df['temp_night'] = df['temperature'].map(lambda x: int(x[(x.find('/')+1):].strip('[ ℃]')))
    return df

# 分割白天和夜间的风力风向
def addDayNightWind(df):
    df['wind_day'] = df['wind'].map(lambda x: x[:x.find('/')].strip(' '))
    df['wind_night'] = df['wind'].map(lambda x: x[(x.find('/')+1):].strip(' '))
    df['wind_dir_day'] = df['wind_day'].map(lambda x: x[:x.find(' ')])
    df['wind_force_day'] = df['wind_day'].map(lambda x: x[(x.find(' ')+1):])
    df['wind_dir_night'] = df['wind_night'].map(lambda x: x[:x.find(' ')])
    df['wind_force_night'] = df['wind_night'].map(lambda x: x[(x.find(' ')+1):])
    return df

# 规范化空值
def formatNan(df):
    cols = df.columns
    for col in cols:
        df[col] = df[col].map(lambda x: np.nan if (x=='')|(x=='-') else x)
    return df

if __name__ == '__main__':
    dateList = [datetime.strftime(x,'%Y%m') for x in pd.date_range(start='2013-01', end='2018-01', freq='M')]
    dfList = list()
    for dt in dateList:
        url = getRequestUrl('yancheng',dt)
        html = getHTMLSoup(url)
        content = html.select('#content > table')[0]
        df = htmlTableToDf(content)
        dfList.append(df)
        print(dt, len(df))
    df = pd.concat(dfList)
    # 规范化数据方便后续调用
    df.columns = ['date','status','temperature','wind']
    df = addDayNightStatus(df)
    df = addDayNightTemp(df)
    df = addDayNightWind(df)
    df = formatNan(df)
    print(df.head())
    print(df.info())
    df.to_csv('../weather.csv', index=False)
