import pandas as pd
import numpy as np

import yfinance as yf
from sklearn import impute
import datetime as dt
import talib

# Data extracts

#Features: Spreads, Xassets, Fin Ratios (extracted from Bloomberg and Capital IQ!)

spreads=pd.read_csv(r'../inputs/spreads.csv');spreads['Date']=pd.to_datetime(spreads['Date'])
xassets=pd.read_excel(r'../inputs/xassets.xlsx')

#Imputing missing values:
imputer=impute.SimpleImputer()
xassets=pd.DataFrame(imputer.fit_transform(xassets.set_index('Date')),columns=xassets.set_index('Date').columns,index=xassets.set_index('Date').index)

dtickers=['BATSLN','DT','RENAUL','SIEGR','TTEFP'] #Debt tickers used to extract values from Capital IQ, Bloomberg and iBoxx
fin_ratios={}

for ticker in dtickers:
    fin_ratios[ticker]=pd.read_excel(r'../inputs/fin_ratios.xlsx',sheet_name=ticker).replace(0,np.nan).replace('NM',np.nan).bfill().ffill().dropna(how='all',axis=1)
    fin_ratios[ticker]['Date']=fin_ratios[ticker].iloc[:,0].apply(lambda x:dt.datetime.strptime(str(int(x[2:3])*3)+'-'+str(x[-4:]),'%m-%Y'))
    fin_ratios[ticker]=fin_ratios[ticker].iloc[:,1:].set_index('Date')
    fin_ratios[ticker].columns=[c for c in map(lambda x:x.split('iq_')[1],fin_ratios[ticker].columns.tolist())]
    
# Extracting Stock Prices from Yahoo Finance!

ytickers=['BATS.L','DTE.DE','RNO.PA','SIE.DE','TTE.PA'] #5 European stocks from different industries
stocks={}

for ticker,ticker_ in zip(ytickers,dtickers):
    stocks[ticker]=yf.download(progress=False,start='2004-01-01',end='2021-08-10',tickers=ticker)
    stocks[ticker]['dret']=stocks[ticker].Close.apply(lambda x:np.log(x))-stocks[ticker].Close.shift(1).apply(lambda x:np.log(x))
    stocks[ticker]=stocks[ticker].dropna()

    #combining prices with spreads,finratios and xassets:
    stocks[ticker]=stocks[ticker].join(xassets).join(spreads[spreads.Ticker==ticker_].drop('Ticker',axis=1).set_index('Date')).join(fin_ratios[ticker_])
    stocks[ticker]=stocks[ticker].bfill().ffill()#missing fin ratios
    stocks[ticker]=stocks[ticker].dropna(how='all',axis=1)
    
# Technical Indicators from TALIB

def TA(df):
    # Cyclical Indicators
    df['HT_DCPERIOD']=talib.HT_DCPERIOD(df.Close)
    df['HT_DCPHASE']=talib.HT_DCPHASE(df.Close)
    df['HT_PHASOR1'],df['HT_PHASOR2']=talib.HT_PHASOR(df.Close)
    df['HT_SINE1'],df['HT_SINE2']=talib.HT_SINE(df.Close)
    df['HT_TRENDMODE']=talib.HT_TRENDMODE(df.Close)

    # Math Operators
    df['ADD']=talib.ADD(df.High,df.Low)
    df['DIV']=talib.DIV(df.High,df.Low)
    df['MAX']=talib.MAX(df.Close,timeperiod=30)
    df['MAXINDEX']=talib.MAXINDEX(df.Close,timeperiod=30)
    df['MIN']=talib.MIN(df.Close,timeperiod=30)
    df['MININDEX']=talib.MININDEX(df.Close,timeperiod=30)
    df['MULT']=talib.MULT(df.High,df.Low)
    df['SUB']=talib.SUB(df.High,df.Low)
    df['SUM']=talib.SUM(df.Close,timeperiod=30)

    # Math Transform
    df['ACOS']=talib.ACOS(df.Close)
    df['ASIN']=talib.ASIN(df.Close)
    df['ATAN']=talib.ATAN(df.Close)
    df['CEIL']=talib.CEIL(df.Close)
    df['COS']=talib.COS(df.Close)
    df['COSH']=talib.COSH(df.Close)
    df['EXP']=talib.EXP(df.Close)
    df['FLOOR']=talib.FLOOR(df.Close)
    df['LN']=talib.LN(df.Close)
    df['LOG10']=talib.LOG10(df.Close)
    df['SIN']=talib.SIN(df.Close)
    df['SINH']=talib.SINH(df.Close)
    df['SQRT']=talib.SQRT(df.Close)
    df['TAN']=talib.TAN(df.Close)
    df['TANH']=talib.TANH(df.Close)

    # Momentum Indicators
    df['ADX']=talib.ADX(df.High,df.Low,df.Close,timeperiod=14)
    df['ADXR']=talib.ADXR(df.High,df.Low,df.Close,timeperiod=14)
    df['APO']=talib.APO(df.Close,fastperiod=12,slowperiod=26,matype=0)
    df['AROONDN'],df['AROONUP']=talib.AROON(df.High,df.Low,timeperiod=14)
    df['AROONOSC']=talib.AROONOSC(df.High,df.Low,timeperiod=14)
    df['BOP']=talib.BOP(df.Open,df.High,df.Low,df.Close)
    df['CCI']=talib.CCI(df.High,df.Low,df.Close,timeperiod=14)
    df['CMO']=talib.CMO(df.Close,timeperiod=14)
    df['DX']=talib.DX(df.High,df.Low,df.Close,timeperiod=14)
    df['MACD'],df['MACDSIGNAL'],df['MACDHIST']=talib.MACDEXT(df.Close,fastperiod=12,fastmatype=0,slowperiod=26,slowmatype=0,signalperiod=9,signalmatype=0)
    df['MFI']=talib.MFI(df.High,df.Low,df.Close,df.Volume,timeperiod=14)
    df['MINUS_DI']=talib.MINUS_DI(df.High,df.Low,df.Close,timeperiod=14)
    df['MINUS_DM']=talib.MINUS_DM(df.High,df.Low,timeperiod=14)
    df['MOM10']=talib.MOM(df.Close,timeperiod=10)
    df['MOM30']=talib.MOM(df.Close,timeperiod=30)
    df['MOM50']=talib.MOM(df.Close,timeperiod=50)
    df['PLUS_DI']=talib.PLUS_DI(df.High,df.Low,df.Close,timeperiod=14)
    df['PLUS_DM']=talib.PLUS_DM(df.High,df.Low,timeperiod=14)
    df['PPO']=talib.PPO(df.Close,fastperiod=12,slowperiod=26,matype=0)
    df['ROC']=talib.ROC(df.Close,timeperiod=10)
    df['ROCP']=talib.ROCP(df.Close,timeperiod=10)
    df['ROCR']=talib.ROCR(df.Close,timeperiod=10)
    df['ROCR100']=talib.ROCR100(df.Close,timeperiod=10)
    df['RSI']=talib.RSI(df.Close,timeperiod=14)
    df['SLOWK'],df['SLOWD']=talib.STOCH(df.High,df.Low,df.Close,fastk_period=5,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
    df['FASTK'],df['FASTD']=talib.STOCHF(df.High,df.Low,df.Close,fastk_period=5,fastd_period=3,fastd_matype=0)
    df['FASTK_'],df['FASTD_']=talib.STOCHRSI(df.Close,timeperiod=14,fastk_period=5,fastd_period=3,fastd_matype=0)
    df['TRIX']=talib.TRIX(df.Close,timeperiod=30)
    df['ULTOSC']=talib.ULTOSC(df.High,df.Low,df.Close,timeperiod1=7,timeperiod2=14,timeperiod3=28)
    df['WILLR']=talib.WILLR(df.High,df.Low,df.Close,timeperiod=14)

    # Overlap Studies
    df['BBU'],df['BBM'],df['BBL']=talib.BBANDS(df.Close,timeperiod=5,nbdevup=2,nbdevdn=2,matype=0)
    df['DEMA10']=talib.DEMA(df.Close,timeperiod=10)
    df['DEMA30']=talib.DEMA(df.Close,timeperiod=30)
    df['EMA5']=talib.EMA(df.Close,timeperiod=5)
    df['EMA10']=talib.EMA(df.Close,timeperiod=10)
    df['EMA30']=talib.EMA(df.Close,timeperiod=30)
    df['HT_TRENDLINE']=talib.HT_TRENDLINE(df.Close)
    df['KAMA10']=talib.KAMA(df.Close,timeperiod=10)
    df['KAMA30']=talib.KAMA(df.Close,timeperiod=30)
    df['MA5']=talib.MA(df.Close,timeperiod=5,matype=0)
    df['MA10']=talib.MA(df.Close,timeperiod=10,matype=0)
    df['MA30']=talib.MA(df.Close,timeperiod=30,matype=0)
    df['MAMA'],df['FAMA']=talib.MAMA(df.Close)
    df['MIDPOINT']=talib.MIDPOINT(df.Close,timeperiod=14)
    df['MIDPRICE']=talib.MIDPRICE(df.High,df.Low,timeperiod=14)
    df['SAR']=talib.SAR(df.High,df.Low,acceleration=0,maximum=0)
    df['SMA5']=talib.SMA(df.Close,timeperiod=5)
    df['SMA10']=talib.SMA(df.Close,timeperiod=10)
    df['SMA30']=talib.SMA(df.Close,timeperiod=30)
    df['T3']=talib.T3(df.Close,timeperiod=5,vfactor=0)
    df['TEMA10']=talib.TEMA(df.Close,timeperiod=10)
    df['TEMA30']=talib.TEMA(df.Close,timeperiod=30)
    df['TRIMA10']=talib.TRIMA(df.Close,timeperiod=10)
    df['TRIMA30']=talib.TRIMA(df.Close,timeperiod=30)
    df['WMA10']=talib.WMA(df.Close,timeperiod=10)
    df['WMA30']=talib.WMA(df.Close,timeperiod=30)

    # Pattern Recognition
    df['CDL2CROWS']=talib.CDL2CROWS(df.Open,df.High,df.Low,df.Close)
    df['CDL3BLACKCROWS']=talib.CDL3BLACKCROWS(df.Open,df.High,df.Low,df.Close)
    df['CDL3INSIDE']=talib.CDL3INSIDE(df.Open,df.High,df.Low,df.Close)
    df['CDL3LINESTRIKE']=talib.CDL3LINESTRIKE(df.Open,df.High,df.Low,df.Close)
    df['CDL3OUTSIDE']=talib.CDL3OUTSIDE(df.Open,df.High,df.Low,df.Close)
    df['CDL3STARSINSOUTH']=talib.CDL3STARSINSOUTH(df.Open,df.High,df.Low,df.Close)
    df['CDL3WHITESOLDIERS']=talib.CDL3WHITESOLDIERS(df.Open,df.High,df.Low,df.Close)
    df['CDLABANDONEDBABY']=talib.CDLABANDONEDBABY(df.Open,df.High,df.Low,df.Close,penetration=0)
    df['CDLADVANCEBLOCK']=talib.CDLADVANCEBLOCK(df.Open,df.High,df.Low,df.Close)
    df['CDLBELTHOLD']=talib.CDLBELTHOLD(df.Open,df.High,df.Low,df.Close)
    df['CDLBREAKAWAY']=talib.CDLBREAKAWAY(df.Open,df.High,df.Low,df.Close)
    df['CDLCLOSINGMARUBOZU']=talib.CDLCLOSINGMARUBOZU(df.Open,df.High,df.Low,df.Close)
    df['CDLCONCEALBABYSWALL']=talib.CDLCONCEALBABYSWALL(df.Open,df.High,df.Low,df.Close)
    df['CDLCOUNTERATTACK']=talib.CDLCOUNTERATTACK(df.Open,df.High,df.Low,df.Close)
    df['CDLDARKCLOUDCOVER']=talib.CDLDARKCLOUDCOVER(df.Open,df.High,df.Low,df.Close,penetration=0)
    df['CDLDOJI']=talib.CDLDOJI(df.Open,df.High,df.Low,df.Close)
    df['CDLDOJISTAR']=talib.CDLDOJISTAR(df.Open,df.High,df.Low,df.Close)
    df['CDLDRAGONFLYDOJI']=talib.CDLDRAGONFLYDOJI(df.Open,df.High,df.Low,df.Close)
    df['CDLENGULFING']=talib.CDLENGULFING(df.Open,df.High,df.Low,df.Close)
    df['CDLEVENINGDOJISTAR']=talib.CDLEVENINGDOJISTAR(df.Open,df.High,df.Low,df.Close,penetration=0)
    df['CDLEVENINGSTAR']=talib.CDLEVENINGSTAR(df.Open,df.High,df.Low,df.Close,penetration=0)
    df['CDLGAPSIDESIDEWHITE']=talib.CDLGAPSIDESIDEWHITE(df.Open,df.High,df.Low,df.Close)
    df['CDLGRAVESTONEDOJI']=talib.CDLGRAVESTONEDOJI(df.Open,df.High,df.Low,df.Close)
    df['CDLHAMMER']=talib.CDLHAMMER(df.Open,df.High,df.Low,df.Close)
    df['CDLHANGINGMAN']=talib.CDLHANGINGMAN(df.Open,df.High,df.Low,df.Close)
    df['CDLHARAMI']=talib.CDLHARAMI(df.Open,df.High,df.Low,df.Close)
    df['CDLHARAMICROSS']=talib.CDLHARAMICROSS(df.Open,df.High,df.Low,df.Close)
    df['CDLHIGHWAVE']=talib.CDLHIGHWAVE(df.Open,df.High,df.Low,df.Close)
    df['CDLHIKKAKE']=talib.CDLHIKKAKE(df.Open,df.High,df.Low,df.Close)
    df['CDLHIKKAKEMOD']=talib.CDLHIKKAKEMOD(df.Open,df.High,df.Low,df.Close)
    df['CDLHOMINGPIGEON']=talib.CDLHOMINGPIGEON(df.Open,df.High,df.Low,df.Close)
    df['CDLIDENTICAL3CROWS']=talib.CDLIDENTICAL3CROWS(df.Open,df.High,df.Low,df.Close)
    df['CDLINNECK']=talib.CDLINNECK(df.Open,df.High,df.Low,df.Close)
    df['CDLINVERTEDHAMMER']=talib.CDLINVERTEDHAMMER(df.Open,df.High,df.Low,df.Close)
    df['CDLKICKING']=talib.CDLKICKING(df.Open,df.High,df.Low,df.Close)
    df['CDLKICKINGBYLENGTH']=talib.CDLKICKINGBYLENGTH(df.Open,df.High,df.Low,df.Close)
    df['CDLLADDERBOTTOM']=talib.CDLLADDERBOTTOM(df.Open,df.High,df.Low,df.Close)
    df['CDLLONGLEGGEDDOJI']=talib.CDLLONGLEGGEDDOJI(df.Open,df.High,df.Low,df.Close)
    df['CDLLONGLINE']=talib.CDLLONGLINE(df.Open,df.High,df.Low,df.Close)
    df['CDLMARUBOZU']=talib.CDLMARUBOZU(df.Open,df.High,df.Low,df.Close)
    df['CDLMATCHINGLOW']=talib.CDLMATCHINGLOW(df.Open,df.High,df.Low,df.Close)
    df['CDLMATHOLD']=talib.CDLMATHOLD(df.Open,df.High,df.Low,df.Close,penetration=0)
    df['CDLMORNINGDOJISTAR']=talib.CDLMORNINGDOJISTAR(df.Open,df.High,df.Low,df.Close,penetration=0)
    df['CDLMORNINGSTAR']=talib.CDLMORNINGSTAR(df.Open,df.High,df.Low,df.Close,penetration=0)
    df['CDLONNECK']=talib.CDLONNECK(df.Open,df.High,df.Low,df.Close)
    df['CDLPIERCING']=talib.CDLPIERCING(df.Open,df.High,df.Low,df.Close)
    df['CDLRICKSHAWMAN']=talib.CDLRICKSHAWMAN(df.Open,df.High,df.Low,df.Close)
    df['CDLRISEFALL3METHODS']=talib.CDLRISEFALL3METHODS(df.Open,df.High,df.Low,df.Close)
    df['CDLSEPARATINGLINES']=talib.CDLSEPARATINGLINES(df.Open,df.High,df.Low,df.Close)
    df['CDLSHOOTINGSTAR']=talib.CDLSHOOTINGSTAR(df.Open,df.High,df.Low,df.Close)
    df['CDLSHORTLINE']=talib.CDLSHORTLINE(df.Open,df.High,df.Low,df.Close)
    df['CDLSPINNINGTOP']=talib.CDLSPINNINGTOP(df.Open,df.High,df.Low,df.Close)
    df['CDLSTICKSANDWICH']=talib.CDLSTICKSANDWICH(df.Open,df.High,df.Low,df.Close)
    df['CDLTAKURI']=talib.CDLTAKURI(df.Open,df.High,df.Low,df.Close)
    df['CDLTASUKIGAP']=talib.CDLTASUKIGAP(df.Open,df.High,df.Low,df.Close)
    df['CDLTHRUSTING']=talib.CDLTHRUSTING(df.Open,df.High,df.Low,df.Close)
    df['CDLTRISTAR']=talib.CDLTRISTAR(df.Open,df.High,df.Low,df.Close)
    df['CDLUNIQUE3RIVER']=talib.CDLUNIQUE3RIVER(df.Open,df.High,df.Low,df.Close)
    df['CDLUPSIDEGAP2CROWS']=talib.CDLUPSIDEGAP2CROWS(df.Open,df.High,df.Low,df.Close)
    df['CDLXSIDEGAP3METHODS']=talib.CDLXSIDEGAP3METHODS(df.Open,df.High,df.Low,df.Close)

    # Price Transform
    df['AVGPRICE']=talib.AVGPRICE(df.Open,df.High,df.Low,df.Close)
    df['MEDPRICE']=talib.MEDPRICE(df.High,df.Low)
    df['TYPPRICE']=talib.TYPPRICE(df.High,df.Low,df.Close)
    df['WCLPRICE']=talib.WCLPRICE(df.High,df.Low,df.Close)

    # Statistic Functions
    df['BETA']=talib.BETA(df.High,df.Low,timeperiod=5)
    df['CORREL10']=talib.CORREL(df.High,df.Low,timeperiod=10)
    df['CORREL30']=talib.CORREL(df.High,df.Low,timeperiod=30)
    df['LINEARREG']=talib.LINEARREG(df.Close,timeperiod=14)
    df['LINEARREG_ANGLE']=talib.LINEARREG_ANGLE(df.Close,timeperiod=14)
    df['LINEARREG_INTERCEPT']=talib.LINEARREG_INTERCEPT(df.Close,timeperiod=14)
    df['LINEARREG_SLOPE']=talib.LINEARREG_SLOPE(df.Close,timeperiod=14)
    df['STDDEV']=talib.STDDEV(df.Close,timeperiod=5,nbdev=1)
    df['TSF']=talib.TSF(df.Close,timeperiod=14)
    df['VAR']=talib.VAR(df.Close,timeperiod=5,nbdev=1)

    # Volatility Indicators
    df['ATR5']=talib.ATR(df.High,df.Low,df.Close,timeperiod=5)
    df['ATR10']=talib.ATR(df.High,df.Low,df.Close,timeperiod=10)
    df['ATR20']=talib.ATR(df.High,df.Low,df.Close,timeperiod=20)
    df['ATR30']=talib.ATR(df.High,df.Low,df.Close,timeperiod=30)
    df['ATR50']=talib.ATR(df.High,df.Low,df.Close,timeperiod=50)
    df['ATR100']=talib.ATR(df.High,df.Low,df.Close,timeperiod=100)
    df['NATR5']=talib.NATR(df.High,df.Low,df.Close,timeperiod=5)
    df['NATR10']=talib.NATR(df.High,df.Low,df.Close,timeperiod=10)
    df['NATR20']=talib.NATR(df.High,df.Low,df.Close,timeperiod=20)
    df['NATR30']=talib.NATR(df.High,df.Low,df.Close,timeperiod=30)
    df['NATR50']=talib.NATR(df.High,df.Low,df.Close,timeperiod=50)
    df['NATR100']=talib.NATR(df.High,df.Low,df.Close,timeperiod=100)
    df['TRANGE']=talib.TRANGE(df.High,df.Low,df.Close)

    # Volume Indicators
    df['AD']=talib.AD(df.High,df.Low,df.Close,df.Volume)
    df['ADOSC']=talib.ADOSC(df.High,df.Low,df.Close,df.Volume,fastperiod=3,slowperiod=10)
    df['OBV']=talib.OBV(df.Close,df.Volume)

    return df

for ticker in ytickers:
    stocks[ticker]=TA(stocks[ticker].copy())

    stocks[ticker]=stocks[ticker].replace(np.inf,np.nan).dropna(how='all',axis=1)

    imputer=impute.SimpleImputer().fit(stocks[ticker])
    stocks[ticker]=pd.DataFrame(imputer.transform(stocks[ticker]),columns=stocks[ticker].columns,index=stocks[ticker].index)