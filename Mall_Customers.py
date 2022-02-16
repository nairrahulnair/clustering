import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv('D:/Datasets/Mall_Customers.csv')

data.head() 
data 
data.shape 
X=data[['Age','Annual Income (k$)','Spending Score (1-100)' ]]
X.head()

## CREATING 3D PLOTS FOR USINF MATPLOTLIB
fig=plt.figure(figsize=(6,6))

ax=fig.add_subplot(111, projection='3d')
ax.scatter(X['Age'],X['Annual Income (k$)'],X['Spending Score (1-100)'], 
           linewidth=1, edgecolors='k',s=200, c=X['Spending Score (1-100)']
           )

plt.show() 



kmeans=KMeans(n_clusters=3,random_state=0).fit(X)
y_pred=kmeans.predict(X)
kmeans.labels_ 
kmeans.cluster_centers_

X['cluster']=y_pred### Adding the cluster column
X.rename(columns={'Annual Income (k$)':'Income','Spending Score (1-100)':'Spend Score'}, inplace=True)

X




df1=X[X.cluster==0]
df2=X[X.cluster==1] 
df3=X[X.cluster==2]
df1 
df2 
df2.describe()
df3.describe()
plt.hist(df3.Income, bins=25, color='red')


fig2=plt.figure(figsize=(6,6))

ax1=fig2.add_subplot(111, projection='3d')
ax1.scatter(df1['Age'],df1['Income'], df1['Spend Score'], linewidth=2, edgecolors='k',s=250)
ax1.scatter(df2['Age'],df2['Income'], df2['Spend Score'], linewidth=2, edgecolors='k',s=250)
ax1.scatter(df3['Age'],df3['Income'], df3['Spend Score'], linewidth=2, edgecolors='k',s=250)

ax1.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], color='red', s=100)
plt.show()
 
### the figure shows that there is an overlap of clusters hence we will 
### figure out the optimum cluster values for the model

def elbow(k_rng):
    
    k_rng=range(1,10)
    sse=[]
    for k in k_rng:
        km=KMeans(n_clusters=k)
        km.fit(X)
        sse.append(km.inertia_)

elbow(sse) 

sse### sum squared error of the model

### plotting the elbow of the sse and k
fig3=plt.figure(figsize=(10,10))
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rng,sse)

#### tuning the model

kmeans2=KMeans(n_clusters=6, random_state=0).fit(X)
y_pred2=kmeans2.predict(X)
kmeans2.inertia_ #### SSe of the model


## amended the cluster column
X['cluster']=y_pred2 
X

df4=X[X.cluster==0]
df5=X[X.cluster==1]
df6=X[X.cluster==2]
df7=X[X.cluster==3]
df8=X[X.cluster==4]
df9=X[X.cluster==5]

X.describe() 
df4.describe() 
df5.describe()
df6.describe()
df7.describe()

y=pd.DataFrame([[55,25,6]])
y
kmeans2.predict(y)
kmeans2.cluster_centers_

fig5=plt.figure(figsize=(15,15))

ax5=fig5.add_subplot(111, projection='3d')

ax5.scatter(df4.Age, df4.Income, df4['Spend Score'], linewidth=3, edgecolors='m', s=250)

ax5.scatter(df5.Age, df5.Income, df5['Spend Score'], linewidth=3, edgecolors='m', s=250)

ax5.scatter(df6.Age, df6.Income, df6['Spend Score'], linewidth=3, edgecolors='k',s=250)

ax5.scatter(df7.Age, df7.Income, df7['Spend Score'], linewidth=3, edgecolors='m', s=250)

ax5.scatter(df8.Age, df8.Income, df8['Spend Score'], linewidth=3, edgecolors='k', s=250)

ax5.scatter(df9.Age, df9.Income, df9['Spend Score'], linewidth=3, edgecolors='k', s=250)

ax5.scatter(kmeans2.cluster_centers_[:,0], kmeans2.cluster_centers_[:,1], kmeans2.cluster_centers_[:,2],
            color='black',s=150
            )
            
ax5.scatter(kmeans2.cluster_centers_[:,3], kmeans2.cluster_centers_[:,4],
            kmeans2.cluster_centers_[:,5], color='black',
            s=150
            )

fig6=plt.figure(figsize=(15,15))





