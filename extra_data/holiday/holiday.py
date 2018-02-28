#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
该接口是按照休息日节假日的工资加倍标准来区分
需按情况手动将节假日前后调休的休息日也改为节假日标记

'''

import urllib, urllib.parse, urllib.request
from bs4 import BeautifulSoup
import re
import csv
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from datetime import *
import json

# 请求第三方接口获取日期休假情况(date字符串格式为YYYYMMDD)
# @return: 返回结果：0：工作日，1：休息日，2：节假日
def getHoliday(dateList):
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

if __name__ == '__main__':
    dateList = pd.date_range(start='2012-01-01', end='2018-01-01', freq='D')
    df = pd.DataFrame({'date':dateList})
    resDict = getHoliday(dateList)
    df['holiday_type'] = df.date.map(lambda x: resDict[x.strftime('%Y%m%d')])
    print(df.head())

    df.to_csv('../holiday_api.csv', index=False)
