#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 250
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme(style='darkgrid')


# In[ ]:


#Dataset


# In[5]:


df = pd.read_excel(r"C:\Users\mange\Downloads\Universities.xlsx")
df


# In[6]:


df.drop(columns=['Univ'], inplace=True)
df


# In[7]:


sns.pairplot(df)


# # Feature Scaling 

# In[8]:


from sklearn.preprocessing import StandardScaler


# In[9]:


sc = StandardScaler()

x = sc.fit_transform(df)
x


# In[ ]:


#Dendrogram with single and complete linkage


# In[10]:


from scipy.cluster import hierarchy


# In[11]:


lk = hierarchy.linkage(x, method='single')
dendrogram = hierarchy.dendrogram(lk)


# In[12]:


lk = hierarchy.linkage(x, method='complete')
dendrogram = hierarchy.dendrogram(lk)
plt.axhline(y = 7, linestyle = '--', color = 'red')


# # Model Building

# In[13]:


from sklearn.cluster import AgglomerativeClustering


# In[14]:


hc = AgglomerativeClustering(n_clusters=2)

ypred = hc.fit_predict(x)
ypred


# In[15]:


df


# In[16]:


df['cluster'] = ypred
df


# In[17]:


df[df['cluster']==0].mean()


# In[18]:


df[df['cluster'] == 1]


# In[ ]:


#Analyzing the clusters


# In[19]:


df.groupby('cluster').agg('mean')


# # 4 Cluster

# In[21]:


lk = hierarchy.linkage(x, method='complete')
dendrogram = hierarchy.dendrogram(lk)
plt.axhline(y = 4, linestyle = '--', color = 'red')


# In[22]:


hc = AgglomerativeClustering(n_clusters=4)
yp = hc.fit_predict(x)
yp


# In[25]:


df = pd.read_excel(r"C:\Users\mange\Downloads\Universities.xlsx")
df['Cluster'] = yp


# In[26]:


df


# In[34]:


df.drop(columns=['Univ'], inplace=True)
df


# In[35]:


df.groupby('Cluster').agg('mean')


# In[36]:


# Hue = for different colur coding

sns.scatterplot(x = df['Top10'], y = df['Expenses'], hue=df['Cluster'], palette='rainbow')


# In[37]:


df[df['Cluster']==2]


# In[38]:


df[df['Cluster'] == 1]


# In[39]:


df[df['Cluster'] == 3]


# In[40]:


df[df['Cluster'] == 0]

