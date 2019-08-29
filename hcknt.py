import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# %matplotlib inline
df=pd.read_csv('trainms.csv')
df.isnull().sum()
df=df.drop(['Country','Gender','state','work_interfere','comments'],axis=1)
df=df.drop(['Timestamp'],axis=1)
df=df.replace(to_replace =['no','No','N','NO'],value =0)
df=df.replace(to_replace =['yes','Yes','Y','YES'],value =1)
df=pd.get_dummies(df,columns=["self_employed","family_history","remote_work","tech_company","benefits","care_options","wellness_program"])
df=df.drop(["benefits_0","benefits_Don't know",'care_options_0','care_options_Not sure','wellness_program_0',"wellness_program_Don't know"],axis=1)
df=df.drop(["family_history_0","remote_work_0","tech_company_0"],axis=1)
df.isnull().sum()
df.dropna()
df=df.drop(['no_employees'],axis=1)
df=df.replace(to_replace =["Don't know"],value =0)
df=df.replace(to_replace =["Very easy"],value =1)
df=df.replace(to_replace =["Somewhat easy"],value =2)
df=df.replace(to_replace =["Somewhat difficult"],value =3)
df=df.replace(to_replace =["Very difficult"],value =4)
df=df.replace(to_replace =["Don't know","Not sure","Maybe",'Some of them'],value =0.5)
df=df.rename(columns={'mental_health_consequence': "mhc", "phys_health_consequence": "phc","phys_health_interview": "phi"})
df['Age']=np.where(df['Age']<=0, df["Age"].median(skipna=True), df['Age'])

labels=df['treatment']
df_train=df.drop(['treatment'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df_train,labels,test_size=0.2)
model=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(model.score(x_test,y_test))

df_test=pd.read_csv('testms.csv')

df_test.isnull().sum()
df_test=df_test.drop(['Country','Gender','state','work_interfere','comments'],axis=1)
df_test=df_test.drop(['Timestamp'],axis=1)
df_test=df_test.replace(to_replace =['no','No','N','NO'],value =0)
df_test=df_test.replace(to_replace =['yes','Yes','Y','YES'],value =1)
df_test=pd.get_dummies(df_test,columns=["self_employed","family_history","remote_work","tech_company","benefits","care_options","wellness_program"])
df_test=df_test.drop(["benefits_0","benefits_Don't know",'care_options_0','care_options_Not sure','wellness_program_0',"wellness_program_Don't know"],axis=1)
df_test=df_test.drop(["family_history_0","remote_work_0","tech_company_0"],axis=1)
df_test.isnull().sum()
df_test.dropna()
df_test=df_test.drop(['no_employees'],axis=1)
df_test=df_test.replace(to_replace =["Don't know"],value =0)
df_test=df_test.replace(to_replace =["Very easy"],value =1)
df_test=df_test.replace(to_replace =["Somewhat easy"],value =2)
df_test=df_test.replace(to_replace =["Somewhat difficult"],value =3)
df_test=df_test.replace(to_replace =["Very difficult"],value =4)
df_test=df_test.replace(to_replace =["Don't know","Not sure","Maybe",'Some of them'],value =0.5)
df_test=df_test.rename(columns={'mental_health_consequence': "mhc", "phys_health_consequence": "phc","phys_health_interview": "phi"})
df_test['Age']=np.where(df_test['Age']<=0, df_test["Age"].median(skipna=True), df_test['Age'])





Predictions=model.predict(df_test)
# Predictions=Predictions.replace(to_replace =[1],value ="Yes")
# Predictions=Predictions.replace(to_replace =[0],value ="No")
df_test['treatment']=Predictions
Final_Dataframe=df_test[['s.no','treatment']]
Final_Dataframe=Final_Dataframe.replace({'treatment':0},{'treatment':"No"})
Final_Dataframe=Final_Dataframe.replace({'treatment':1},{'treatment':"Yes"})

Final_Dataframe.to_csv('Submission_4.csv',index=False)
