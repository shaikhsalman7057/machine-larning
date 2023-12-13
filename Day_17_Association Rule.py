#!/usr/bin/env python
# coding: utf-8

# # Association Rules
# 
# - Association Rules Analysis has become familiar for analysis in the retail industry. It is also called Market Basket Analysis terms. This analysis is also used for advice. Personal recommendations in applications such as Spotify, Netflix, and Youtube can be given as examples. 
# 
# - Association Rules are derived to understand which products go together.
# 
# - Once we get these types of association rules between various products, we can solve multiple business problem statements such as:
#     1. Products to stock
#     2. Promotion on various products
#     3. Implementing strategies to arrange the products in store.
#     4. Giving extra offers on products which are not getting sold.
#     5. Building strategies to improve the customer feedbacks.
# ![image-4.png](attachment:image-4.png)

# # Apriori Algorithm
# 
# - The Apriori Algorithm, used for the first phase of the Association Rules, is the most popular and classical algorithm in the frequent old parts.
# - Apriori algorithm is a classical approach to find frequent patterns and highly related products.
# - The goal is to find combinations of products that are often bought together, which we call frequent itemsets. The technical term for the domain is Frequent Itemset Mining.
# 
# **The importance of Association rule is determined by three metrics:**
# 
# **1.Support:This measure gives an idea of how frequent an itemset is in all the transactions.**
# ![image-3.png](attachment:image-3.png)
# 
# **2.Confidence: This measure defines the likeliness of occurrence of consequent on the cart given that the cart already has the antecedents.**
# ![image-4.png](attachment:image-4.png)
# ![image-2.png](attachment:image-2.png)
# Total transactions = 100. 10 of them have both milk and toothbrush, 70 have milk but no toothbrush and 4 have toothbrush but no milk.
# -  Confidence for {Toothbrush} → {Milk} will be 10/(10+4) = 0.7
# - Looks like a high confidence value. But we know intuitively that these two products have a weak association and there is something misleading about this high confidence value. Lift is introduced to overcome this challenge.
# 
# **3. Lift: Lift tells you how strong the association rule is.**
# 
# ![image-5.png](attachment:image-5.png)
# 
# - Lift : (10/4)/70 = 0.035
# 
# **4. Leverage: With and without item A is in the transaction, mow much it affect item B?** 
# - Leverage computes the probability of A and B occurring together and the frequency that would be expected if A and B were independent.
# - Leverage is similar to lift but easier to interpret since it ranges from -1 to 1 while lift ranges from 0 to infinity.
# - A leverage value of 0 indicates independence.
# ![image-6.png](attachment:image-6.png)
# 
# **5. Conviction: Conviction helps to judge if the rule happened to be there by chance or not.**
# - A high conviction value means that the consequent is highly dependent on the antecedent (A). It can be interpreted as lift.
# - If items are independent, the conviction is 1.
# ![image-7.png](attachment:image-7.png)

# **STEPS INVOLVED IN APRIORI ALGORITHM:**
# 1. Compute the support value for each item:
#     - The support is simply the number of transactions in which a specific product (or combination of products) occurs.
# 2. Deciding the support threshold
#     - Selection of support threshold depends on domain knowledge and the dataset.
# 3. Selecting the one item set based on the support value.
# 4. Selecting two item set:
#     - The next step is to do the same analysis, but now using pairs of products instead of individual products.
# 5. Repeat the same step for larger sets.
# 6. Generate association rule and calculate confidence.
# 7. Compute lift ratio.

# In[2]:


#Install required library
get_ipython().system('pip install mlxtend')


# In[20]:


#Import Libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules,apriori
import warnings
warnings.filterwarnings("ignore")


# # Supermarket (DataSet)

# In[4]:


df = pd.read_csv("https://raw.githubusercontent.com/aishwaryamate/Datasets/main/Supermarket.csv",index_col=0)
df


# In[6]:


df.iloc[[0]]


# In[7]:


txt = 'Milk,Bread,sauce'
txt
txt.split(',')


# In[8]:


# Function is used for the Splitting the data .

def txt_split(txt):
    return txt.split(',')


# In[9]:


data = df['Products'].apply(txt_split)


# In[11]:


data


# In[12]:


# Transaction encoder is used for getting the unique products form the Dataset 


from mlxtend.preprocessing import TransactionEncoder


# In[13]:


te = TransactionEncoder()
encoded_df = te.fit_transform(data)


# In[14]:


encoded_df


# In[15]:


te.columns_


# In[16]:


data = pd.DataFrame(encoded_df,columns=te.columns_)
data


# In[17]:


data.replace(True,1,inplace=True)
data.replace(False,0,inplace=True)
data


# In[21]:


scores = apriori(data,min_support=0.2, use_colnames=True)
scores


# In[22]:


rules = association_rules(scores)
rules


# In[26]:


rules.sort_values(by = 'confidence', ascending=False)


# In[ ]:




