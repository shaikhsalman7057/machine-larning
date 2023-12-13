#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 250
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Load data


# In[2]:


df = pd.read_csv(r"C:\Users\mange\Downloads\Mall_Data.csv", index_col=0)
df


# In[3]:


#Data visualization
sns.scatterplot(x= df['Annual Income (k$)'], y = df['Spending Score (1-100)'])


# In[5]:


#Standardization
from sklearn.preprocessing import StandardScaler


# In[6]:


sc = StandardScaler()

x = sc.fit_transform(df)
x


# In[ ]:


#Model Building


# In[7]:


from sklearn.cluster import KMeans


# In[8]:


df.head()


# In[9]:


km = KMeans()
yp = km.fit_predict(x)
yp


# In[10]:


len(yp)


# In[11]:


#cluster centre
km.cluster_centers_


# In[12]:


x


# In[13]:


#Visualize the clusters
x[:,0]


# In[14]:


x[:,1]


# In[15]:


plt.scatter(x[:,0],x[:,1], c = yp, cmap='rainbow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], c = 'yellow', s = 100)


# In[16]:


km.inertia_


# # Elbow graph 

# In[18]:


wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(x)
    wcss.append(km.inertia_)


# In[19]:


wcss


# In[20]:


plt.plot(range(1,11),wcss)
plt.axvline(x = 5, linestyle = '--', color = 'red')
plt.xlabel('K-Values')
plt.ylabel('WCSS')
plt.title('Elbow Graph')


# # Final Model

# In[21]:


km = KMeans(n_clusters=5)
ypred = km.fit_predict(x)
ypred


# In[ ]:


#Visualize the clusters


# In[22]:


km.cluster_centers_


# In[23]:


plt.scatter(x[:,0],x[:,1], c = ypred, cmap='rainbow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1], c = 'yellow',s = 200)


# # Evaluation of Kmeans
# 
# <code> **STEP 1:** </code>
# ![1.png](attachment:1.png)
# 

# <code>**STEP 2**</code>
# ![2..png](attachment:2..png)

# <code>**STEP 3**</code>
# ![3..png](attachment:3..png)

# <code>**STEP 4:** </CODE>
# ![4..png](attachment:4..png)

# In[ ]:


#Silhouette score


# In[24]:


from sklearn.metrics import silhouette_score


# In[25]:


silhouette_score(x,ypred)


# In[26]:


#Analyzing clusters
df['cluster'] = ypred
df


# In[27]:


df[df['cluster'] == 0].mean()
df[df['cluster'] == 1].mean()


# In[28]:


df.groupby('cluster').agg('mean')


# In[29]:


df['cluster'].value_counts()


# In[ ]:




