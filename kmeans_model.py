#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib notebook


# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[4]:


# Upload your data as a csv file and load it as a data frame 
dataset = pd.read_csv("data.csv" , encoding= 'unicode_escape').dropna()
dataset.head()
dataset.describe()
dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])


# In[5]:


dataset['command_price'] = dataset.Quantity * dataset.UnitPrice
dataset['command_price'] = dataset['command_price'].astype(int)
dataset['InvoiceDate'] = dataset['InvoiceDate'].dt.to_period('M')
dataset['InvoiceDate']
#dataset


# In[6]:


#selected_atributes = [ 'InvoiceNo' , 'StockCode' , 'CustomerID' , 'Country' , 'InvoiceDate' , 'command_price']
selected_atributes = [ 'InvoiceNo' , 'StockCode' , 'CustomerID' , 'Country' , 'command_price']


# In[7]:


#countries = dict((x, ind) for ind , x in enumerate(dataset.Country.unique()) )

#countries 
#rcountries
#dataset.Country.unique()


# In[8]:


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
countries  = dict((x, ind) for ind , x in enumerate(dataset.Country.unique()) )
rcountries = dict(( x, ind) for ind , x in countries.items() )
dataset['Country'] = dataset['Country'].map(countries)


# In[9]:


X = dataset[selected_atributes]
X


# In[10]:


X.isnull().sum()


# In[11]:


# Visualize the correlation your data and identify variables for further analysis
g = sns.PairGrid(X)
g.map(sns.scatterplot)


# In[12]:


# def find_best_clusters(df, maximum_K):
    
#     clusters_centers = []
#     k_values = []
    
#     for k in range(1, maximum_K):
        
#         kmeans_model = KMeans(n_clusters = k)
#         kmeans_model.fit(df)
        
#         clusters_centers.append(kmeans_model.inertia_)
#         k_values.append(k)
        
    
#     return clusters_centers, k_values


# In[13]:


#find_best_clusters(X, 12)
X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.3, random_state=42)


# In[14]:


# clusters_centers, k_values = find_best_clusters(X, 12)


# # In[15]:


# def generate_elbow_plot(clusters_centers, k_values):
    
#     figure = plt.subplots(figsize = (12, 6))
#     plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
#     plt.xlabel("Number of Clusters (K)")
#     plt.ylabel("Cluster Inertia")
#     plt.title("Elbow Plot of KMeans")
#     plt.show()


# # In[16]:


# generate_elbow_plot(clusters_centers, k_values)


# # In[17]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

scaled_data = scaler.transform(X_train)
scaled_data


# In[18]:


kmeans_model = KMeans(n_clusters = 3)

kmeans_model.fit(scaled_data)


# In[19]:


X_train["clusters"] = kmeans_model.labels_

X_train.head()


# In[26]:


# plt.scatter(X_train["StockCode"], 
#             X_train["command_price"], 
#             c = X_train["clusters"])
# plt.scatter(X_train["Country"], 
#             X_train["CustomerID"],
#             c = X_train["clusters"])
#get_ipython().run_line_magic('matplotlib', 'widget')
fig = plt.figure()
ax = fig.add_subplot( 111 , projection='3d')
ax.scatter(X_train["Country"], 
            X_train["CustomerID"],
            X_train["StockCode"], 
            c = X_train["clusters"])
plt.show()
#ee.show()


# In[ ]:


#y_pred = kmeans_model.predict(X_test)


# In[ ]:


#y_pred
#print( y_pred )


# In[ ]:


#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

