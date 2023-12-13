#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 250
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r"C:\Users\mange\Downloads\Wholesale.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
df


# In[3]:


# It is used to check weather the outliers are present or not 

df.describe()


# In[4]:


df.boxplot()


# In[5]:


sns.pairplot(df)


# In[10]:


#Feature scaling :- Converting all  the data to same scale i.e. nothing but standardizattion.


from sklearn.preprocessing import StandardScaler


# In[11]:


# sc = StandardScaler() For Standardization.


sc = StandardScaler()

x = sc.fit_transform(df)
x


# # Model Building

# In[12]:


from sklearn.cluster import DBSCAN


# In[13]:


db = DBSCAN(eps=0.5, min_samples=3)

y = db.fit_predict(x)                # Calculate the values and store in the y and predict the no of clusters i.e _predict(x)
y


# In[23]:


# Do the Rainbow colour coding ON the basis of Y 

plt.scatter(x[:,0] , x[:,4] , c = y , cmap = 'rainbow')


# In[24]:


from sklearn.metrics import silhouette_score


# In[26]:


# Silhouette_Score = 0 Bad   , Overlapping
#                  = 1 Good
silhouette_score(x,y)


# # Hyperparameter tuning

# # Selecting Min_samples:
# 1. Number of columns + 1
# 2. Number of columns * 2
# 3. Always keep min_sample at least 3

# # Finding out the best eps value(K-dist Plot)
# - This technique calculates the average distance between each point and its k nearest neighbors, where k is the MinPts value you selected. 
# - The average k-distances are then plotted in ascending order on a k-distance graph. You’ll find the optimal value for ε at the point of maximum curvature (i.e. where the graph has the greatest slope).

# In[27]:


from sklearn.neighbors import NearestNeighbors


# In[32]:


neigh = NearestNeighbors(n_neighbors=3).fit(x)


# In[33]:


# index = index no of those data points.
# D     = nearest Distances.


d,index = neigh.kneighbors(x)


# In[35]:


# We only need the 1st nearest Value to plot the (K-dist Plot).

d


# In[36]:


# Take all the records from the column no 1.

distance = np.sort(d[:,1])
distance


# In[38]:


plt.plot(distance)
plt.xlabel('Distance')
plt.ylabel('EPS')
plt.title('K-Distance Graph')

# After plotting the graph plt. axh-line

plt.axhline(y=2, linestyle = '--', color = 'red')


# # Evaluate the model

# In[39]:


# From the graph we get to know the Eps Value.


db = DBSCAN(eps=2,min_samples=3)
yp = db.fit_predict(x)
yp


# In[43]:


# We can use any of the columns on the X and Y axis.


plt.scatter(x[:,1],x[:,2],c = yp, cmap='viridis')


# In[44]:


# It is close to 1 so the model is good.

silhouette_score(x,yp)


# In[ ]:


#Analyze the clusters


# In[45]:


df['Cluster'] = yp
df


# In[46]:


df[df['Cluster'] == -1]


# In[47]:


df.groupby('Cluster').agg('mean')


# In[ ]:




