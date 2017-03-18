import numpy as np 
import pandas as pd 
from pandas import DataFrame
from math import log


#首先引入数据集,用txt文档    we input data set as a .txt file
def Getdata(filename):
    dataset=open(filename)
    Dataset=[]
    
    for line in dataset:
        line=line.strip()
        lineinfo=line.split('\t')
        Dataset.append(lineinfo)
    Dataset=pd.DataFrame(Dataset)
    return Dataset

#计算总熵    calculate entropy
def Wholeshang(dataset):
    numberofcolumns=len(np.array(dataset)[0])
    numberofrows=len(dataset)
    lastcolumns=np.array(dataset.icol(-1))    #这时是一个数组
    theset=set(lastcolumns)
    wholeset=list(theset)
    numberofwholeset=len(wholeset)
    countnumber=np.zeros(numberofwholeset)
    for j in range(numberofwholeset):
        for i in range(numberofrows):
            if lastcolumns[i]==wholeset[j]:
                countnumber[j]+=1
    sumnumber=countnumber.sum()
    percentnumber=countnumber/sumnumber
    for i in range(numberofwholeset):
        percentnumber[i]=log(percentnumber[i],2)*percentnumber[i]*-1
    wholeshang=percentnumber.sum()
    return wholeshang

#将数据分割成多个数组  split data set to lists
def Splitfactor(dataset):
    numberofcolumns=len(np.array(dataset)[0])
    numberofrows=len(dataset)
    splitdata=[]
    for i in range(numberofcolumns):
        aimcolumns=np.array(dataset.icol(i))
        splitdata.append(aimcolumns)
    return splitdata

#这个函数可以计算任意给定一个数组的熵   Given a list, this function can calculate its entropy
def calshang(datalist):
    listlong=len(datalist)
    aimset=set(datalist)
    compareset=list(aimset)
    aimsetlong=len(aimset)
    countnumber=np.zeros(aimsetlong)
    for j in range(aimsetlong):
        comparenumber=compareset[j]
        for i in range(listlong):
            if datalist[i]==comparenumber:
                countnumber[j]+=1
    sumnumber=countnumber.sum()
    percentnumber=countnumber/sumnumber
    for i in range(aimsetlong):
        percentnumber[i]=log(percentnumber[i],2)*percentnumber[i]*-1
    resultshang=percentnumber.sum()
    return resultshang
        
#用计算熵的函数计算每一个数据列的熵   calculate conditional entropy of each features 
def Factorshang(dataset,n):
    aimlist=dataset[n]
    resultset=dataset[-1]
    listlong=len(aimlist)
    theset=set(aimlist)
    Theset=list(theset)
    numberoftheset=len(Theset)
    calset=[]
    for j in range(numberoftheset):
        countnumber=0
        factorshang=0
        for i in range(listlong):
            if aimlist[i]==Theset[j]:
                calset.append(resultset[i])
                countnumber+=1
        factorshang+=countnumber/listlong*calshang(calset)
    return factorshang

#找到各个features 的熵
def factor_entropy_list(dataset):
    Factor_entropy_list=[]
    for i in range(11):
        Factor_entropy_list.append(Factorshang(dataset,i))
    return Factor_entropy_list
        

        

rawdata=Getdata('/Users/longxiaoyi/Desktop/123.txt')
splited_data=Splitfactor(rawdata)
Total_entropy=Wholeshang(rawdata)
Factor_entropy_list=factor_entropy_list(splited_data)

        
    
        
        
        
        


    


