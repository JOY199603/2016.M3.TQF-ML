import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def Getdata(filename):
    dataset=open(filename)
    Dataset=[]
    
    for line in dataset:
        line=line.strip()
        lineinfo=line.split('\t')
        Dataset.append(lineinfo)
    Dataset=pd.DataFrame(Dataset)
    return Dataset

dataset=Getdata('/Users/longxiaoyi/Desktop/123.txt')

featurescolumns=[0,1,2,3,4,5,6,7,8,9,10]
x_data=dataset[featurescolumns]
y_data=dataset[11]
linreg = LinearRegression()  
model=linreg.fit(x_data, y_data)

print(linreg.intercept_)
print(linreg.coef_) 