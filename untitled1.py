import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# %matplotlib inline
df=pd.read_csv('trainms.csv')
df=df.drop(['Country','Gender','state','comments'],axis=1)
df=df.drop(['Timestamp'],axis=1)
df=df.replace(to_replace =['no','No','N','NO'],
                 value =0)
df=df.replace(to_replace =['yes','Yes','Y','YES'],
value =1)
df=df.replace(to_replace =["Very easy"],
                 value =.25)
df=df.replace(to_replace =["Somewhat easy"],
                 value =.5)
df=df.replace(to_replace =["Somewhat difficult"],
                 value =.75)
df=df.replace(to_replace =["Very difficult"],
                 value =1)
df=df.replace(to_replace =["Don't know","Not sure","Maybe",'Some of them'],
                 value =0.5)
df=df.replace(to_replace =["Never"],
                 value =0)
df=df.replace(to_replace =["NA"],
                 value =.5)
df=df.replace(to_replace =["Often"],
                 value =1)
df=df.replace(to_replace =["Rarely"],
                 value =.25)
df=df.replace(to_replace =["Sometimes"],
                 value =.75)
df['Age']=np.where(df['Age']<=0 , df["Age"].median(skipna=True), df['Age'])
df['Age']=np.where(df['Age']>=80 , df["Age"].median(skipna=True), df['Age'])
df=df.drop(['no_employees'],axis=1)
df.dropna()
df["self_employed"].fillna(0.5, inplace = True)
df["work_interfere"].fillna(0.5, inplace = True)


labels=df['treatment']
df_train=df.drop(['treatment'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df_train,labels,test_size=0.2)
model=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
model.fit(x_train,y_train)
prediction=model.predict(x_test)


df1=pd.read_csv('testms.csv')
df1=df1.drop(['Country','Gender','state','comments'],axis=1)
df1=df1.drop(['Timestamp'],axis=1)
df1=df1.replace(to_replace =['no','No','N','NO'],
                 value =0)
df1=df1.replace(to_replace =['yes','Yes','Y','YES'],
value =1)
df1=df1.replace(to_replace =["Very easy"],
                 value =.25)
df1=df1.replace(to_replace =["Somewhat easy"],
                 value =.5)
df1=df1.replace(to_replace =["Somewhat difficult"],
                 value =.75)
df1=df1.replace(to_replace =["Very difficult"],
                 value =1)
df1=df1.replace(to_replace =["Don't know","Not sure","Maybe",'Some of them'],
                 value =0.5)
df1=df1.replace(to_replace =["Never"],
                 value =0)
df1=df1.replace(to_replace =["NA"],
                 value =.5)
df1=df1.replace(to_replace =["Often"],
                 value =1)
df1=df1.replace(to_replace =["Rarely"],
                 value =.25)
df1=df1.replace(to_replace =["Sometimes"],
                 value =.75)
df1['Age']=np.where(df1['Age']<=0 , df1["Age"].median(skipna=True), df1['Age'])
df1=df1.drop(['no_employees'],axis=1)
df1.dropna()
df1["self_employed"].fillna(0.5, inplace = True)
df1["work_interfere"].fillna(0.5, inplace = True)
df1["work_interfere"].values.tolist()




Predictions=model.predict(df1)
# Predictions=Predictions.replace(to_replace =[1],value ="Yes")
# Predictions=Predictions.replace(to_replace =[0],value ="No")
df1['treatment']=Predictions
Final_Dataframe=df1[['s.no','treatment']]
Final_Dataframe=Final_Dataframe.replace({'treatment':0},{'treatment':"No"})
Final_Dataframe=Final_Dataframe.replace({'treatment':1},{'treatment':"Yes"})

Final_Dataframe.to_csv('Submission_1.csv',index=False)