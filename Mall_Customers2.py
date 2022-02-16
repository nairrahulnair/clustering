import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import plotly.express as px 
from pandas.plotting import parallel_coordinates


data=pd.read_csv('D:/Datasets/Mall_Customers.csv')

data.shape

data.head() 

data.rename(columns={'Annual Income (k$)':'Income','Spending Score (1-100)':'Spend Score'}, inplace=True)

data 

X=data[['Age','Income','Spend Score']]

X

scaler=MinMaxScaler()
X=scaler.fit_transform(X)

X 
X=pd.DataFrame(X)
X.rename(columns={0:"Age",1:"Income",2:"Spend Score"}, inplace=True)
X 
fig1=plt.figure(figsize=(15,15))
ax=fig1.add_subplot(111, projection='3d')
plt.xlabel('Age')
plt.ylabel('Income')


ax.scatter(X['Age'],X['Income'],X['Spend Score'],s=55, linewidths=4 ,c=X['Spend Score'], cmap='rainbow')

## Finding the optimum K value using Elbow method
k_range=range(1,20)
sse=[]

for k in k_range:
    km=KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
    
fig2=plt.figure(figsize=(10,10))
plt.xlabel('K value')    
plt.ylabel('SSE')

plt.plot(k_range, sse)

### The plot tells that the optimum value is around 6-7, we choose 7 and proceed 

kmeans2=KMeans(n_clusters=6).fit(X)
y_pred2=kmeans2.predict(X)

kmeans2.inertia_ 
kmeans2.labels_ 

X['Cluster']=kmeans2.labels_ 

df1=X[X['Cluster']==0]
df2=X[X.Cluster==1]
df3=X[X.Cluster==2]
df4=X[X.Cluster==3]
df5=X[X.Cluster==4]
df6=X[X.Cluster==5]


fig2=plt.figure(figsize=(15,15))
ax1=fig2.add_subplot(111, projection='3d')

ax1.scatter(df1.Age, df1.Income, df1['Spend Score'], s=250, edgecolors='k', linewidths=1)
ax1.scatter(df2.Age, df2.Income, df2['Spend Score'], s=250, edgecolors='k', linewidths=1)
ax1.scatter(df3.Age, df3.Income, df3['Spend Score'], s=250, edgecolors='k', linewidths=1)
ax1.scatter(df4.Age, df4.Income, df4['Spend Score'], s=250, edgecolors='k', linewidths=1)

ax1.scatter(df5.Age, df5.Income, df5['Spend Score'], s=250, edgecolors='k', linewidths=1)

ax1.scatter(df6.Age, df6.Income, df6['Spend Score'], s=250, edgecolors='k', linewidths=1)

#ax1.scatter(df7.Age, df7.Income, df7['Spend Score'], s=250, edgecolors='k', linewidths=1)


### printing parallel coordinates for insights

fig3=plt.figure(figsize=(20,8))
parallel_coordinates(df1, "Cluster", colormap="prism")

fig4=plt.figure(figsize=(20,8))
parallel_coordinates(df2,"Cluster", colormap='terrain')

fig5=plt.figure(figsize=(20,8))
parallel_coordinates(df3, "Cluster", colormap='flag')

fig6=plt.figure(figsize=(20,8))
parallel_coordinates(df4, "Cluster", colormap='flag')

fig7=plt.figure(figsize=(20,8))
parallel_coordinates(df5, "Cluster", colormap='flag')

fig8=plt.figure(figsize=(20,8))
parallel_coordinates(df6, "Cluster", colormap='flag')







    
    
    
    
    
    


