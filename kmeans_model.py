#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib notebook


# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[3]:


# Upload your data as a csv file and load it as a data frame 
dataset = pd.read_csv("data.csv" , encoding= 'unicode_escape').dropna()
dataset.head()
dataset.describe()


# In[4]:


# producing a column 
dataset['command_price'] = dataset.Quantity * dataset.UnitPrice
dataset['command_price'] = dataset['command_price'].astype(int)
dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])
dataset['InvoiceDate'] = dataset['InvoiceDate'].dt.to_period('M')


# In[5]:


# select the features 
selected_atributes = [ 'InvoiceNo' , 'StockCode' , 'CustomerID' , 'Country' , 'command_price']


# In[6]:


# clean data and formating 
dataset = dataset[ dataset['CustomerID'].notna() ] 
dataset = dataset[ dataset['StockCode'].notna() ] 
dataset = dataset[ dataset['Quantity'] > 0 ] 
dataset = dataset[ dataset['UnitPrice'] > 0 ] 

dataset['CustomerID'] = dataset['CustomerID'].astype(int)
dataset['InvoiceNo'] = dataset.InvoiceNo.astype(str)
dataset['InvoiceNo'] = dataset.InvoiceNo.str.extract(r'(\d+)', expand=True).astype(int)
dataset['StockCode'] = dataset.StockCode.astype(str)
dataset['StockCode'] = dataset.StockCode.str.extract(r'(\d+)', expand=True)
dataset= dataset[dataset['StockCode'].isnull() == False]

dataset['StockCode'] = dataset.StockCode.astype(int)



# countries = {}
# for ind , x in enumerate(dataset.Country.unique()):
#     countries[x]=ind
countries =  dict((x, ind) for ind , x in enumerate(dataset.Country.unique()) )
rcountries =  dict(( x, ind) for ind , x in countries.items() )
dataset['Country'] = dataset['Country'].map(countries)


# In[7]:


X = dataset[selected_atributes]
# X


# In[8]:


# X.isnull().sum()


# In[9]:


# Visualize the correlation your data and identify variables for further analysis
# g = sns.PairGrid(X)
# g.map(sns.scatterplot)


# In[10]:


# def find_best_clusters(df, maximum_K):
    
#     clusters_centers = []
#     k_values = []
    
#     for k in range(1, maximum_K):
        
#         kmeans_model = KMeans(n_clusters = k)
#         kmeans_model.fit(df)
        
#         clusters_centers.append(kmeans_model.inertia_)
#         k_values.append(k)
        
    
#     return clusters_centers, k_values


# In[11]:


# get training data 
X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.3, random_state=42)


# In[12]:


# get optimal k number of clusters
# clusters_centers, k_values = find_best_clusters(X, 12)


# In[13]:


# function to visualise the optimal k data
# def generate_elbow_plot(clusters_centers, k_values):
    
#     figure = plt.subplots(figsize = (12, 6))
#     plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
#     plt.xlabel("Number of Clusters (K)")
#     plt.ylabel("Cluster Inertia")
#     plt.title("Elbow Plot of KMeans")
#     plt.show()


# In[14]:


#show the plots of optimal k
# generate_elbow_plot(clusters_centers, k_values)


# In[15]:


# transform to numeric data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

scaled_data = scaler.transform(X_train)
scaled_data


# In[16]:


# from the plot we see that k=3 is the optimal k 
# train kmeans model
kmeans_model = KMeans(n_clusters = 3)

kmeans_model.fit(scaled_data)


# In[17]:


# get labels of model

X_train["clusters"] = kmeans_model.labels_

X_train.head()


# In[18]:


# plt.scatter(X_train["Country"], 
#             X_train["CustomerID"],
#             c = X_train["clusters"])


# In[19]:


# plt.scatter(X_train["StockCode"], 
#             X_train["command_price"], 
#             c = X_train["clusters"])
# plt.scatter(X_train["Country"], 
#             X_train["CustomerID"],
#             c = X_train["clusters"])

## show the 3d plot of cluster
# get_ipython().run_line_magic('matplotlib', 'widget')
fig = plt.figure()
ax = fig.add_subplot( 111 , projection='3d')
ax.scatter(X_train["Country"], 
            X_train["CustomerID"],
            X_train["StockCode"], 
            c = X_train["clusters"])
plt.show()
#ee.show()


# In[20]:


# y_pred = kmeans_model.predict(X_test)


# In[21]:


# y_pred
# print( y_pred )


# In[ ]:




