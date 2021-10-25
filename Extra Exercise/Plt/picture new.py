# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:30:58 2021

@author: tyx
"""

import pandas as pd
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Data_new.csv')

for i in range(0,183):
    data_loop=data.loc[i:i]
    for index,row in data_loop.iterrows():
        #r1:current real r-value, r1=RPR/100
        #r2:target r-value, when r/RER-1=0, r2=RER
        #r3:highest-penalty r-value, when r/RER-1=0.03, r3=1.03*RER
        #for index,row in data.iterrows():
        #get r1&r2&r3,three revenue 
        Id=data.iloc[index,1]
        r1=data.iloc[index,5]
        cur_rev=data.iloc[index,9]
        r2=data.iloc[index,6]
        tar_rev=(1/(1-r2))
        r3=1.03*(data.iloc[index,6])
        pen_rev=(1/(1-r3)*(1-0.03*(data.iloc[index,2])))
        #jugde the order of r1,r2,r3
        if r1<r2:
            x=[]
            x=[r1,r2,r3]
            y=[]
            y=[cur_rev,tar_rev,pen_rev]
        elif r1>r3:
            x=[]
            x=[r2,r3,r1]
            y=[]
            y=[tar_rev,pen_rev,cur_rev]
        else:
            x=[]
            x=[r2,r1,r3]
            y=[]
            y=[tar_rev,cur_rev,pen_rev]       
        line=plt.subplot(183,1,index+1)
        #plt.subplot(1,1,1)
        line.plot(x,y)
        plt.title(Id)
        plt.xlabel('r-value')
        plt.ylabel('Revenue')
      
      
