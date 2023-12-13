#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries :

# Importing pandas, numpy, matplotlib.pyplot, and seaborn to enable data manipulation, numerical operations, and data visualization

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data Inspection :

# Data inspection involves examining and cleaning datasets to understand their characteristics, identify patterns, and assess data quality. It includes tasks such as exploratory data analysis, feature inspection, and visualization to ensure reliable and meaningful analysis.

# In[2]:


dataset = pd.read_csv('Mall_Customers.csv')


# In[3]:


dataset.head()


# In[4]:


dataset['Gender'] = dataset['Gender'].map({'Female': '0', 'Male': '1'})
dataset.head()


# In[5]:


dataset.shape


# In[6]:


dataset.columns


# In[7]:


dataset.info()


# Selecting columns in the dataset that have data types of 'int64' or 'float64' 

# In[8]:


dataset.select_dtypes(include = ['int64', 'float64']).columns


# ### Statical Info of Dataset :

# 
# Statistical information refers to data that has been analyzed and summarized using statistical methods. It includes measures such as mean, median, mode, standard deviation, and other descriptive statistics. These metrics provide a quantitative summary of a dataset, revealing central tendencies, variability, and distribution patterns. 

# In[9]:


dataset.describe()


# In[10]:


dataset.isnull().values.any()


# In[11]:


dataset.head()


# Calculating the correlation matrix of the dataset and creating a heatmap.The heatmap visually represents the correlation matrix of the dataset, using colors to show the strength and direction of correlations. Darker shades indicate stronger correlations, while annotations display exact correlation coefficients.

# In[12]:


corr_matrix = dataset.corr()

plt.figure(figsize = (5, 3))
ax = sns.heatmap(corr_matrix,
                 annot = True, 
                 cmap = 'coolwarm')


# Making copy of original dataset

# In[13]:


dataframe = dataset


# ### Standardizing Dataset :

# 
# Standardizing a dataset involves transforming its features so that they have a mean of 0 and a standard deviation of 1. This process is often referred to as "feature scaling" or "data normalization." Standardization is a common preprocessing step in machine learning and statistics.

# 
# Importing the StandardScaler class from scikit-learn. Creating a scaler instance (sc). Standardizing the dataset using fit_transform(), centering features around zero and scaling them to have a standard deviation of one.

# In[14]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(dataset)


# In[15]:


dataset


# ### Clustering :

# 
# Clustering is an unsupervised learning method grouping similar data points. It identifies inherent patterns without predefined outcomes, aiming to form clusters where data points within the same group share similarities.
# 
# 

# ### KMeans :

# 
# KMeans is a partitioning clustering algorithm that divides a dataset into K clusters based on feature similarity. It iteratively assigns data points to clusters, updating centroids until convergence. Sensitive to initial centroid placement, it commonly uses methods like k-means++ for improved convergence. Widely applied in tasks like image segmentation and customer segmentation.

# In[16]:


from sklearn.cluster import KMeans


# Creating a loop to run the k-means clustering algorithm for different numbers of clusters (1 to 19). For each iteration, calculating the within-cluster sum of squares (WCSS) and storing the values in a list (wcss). After the loop, plotting the WCSS values against the number of clusters to find the optimal cluster number using the elbow method. The resulting plot is displayed with labels for clarity.

# In[18]:


wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 15), wcss, 'bx-')
plt.title('The Elbow Methode')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()


# Importing the KElbowVisualizer class from the yellowbrick.cluster module and creating a KMeans model with a random state of 1. Initializing a KElbowVisualizer instance with a range of clusters from 1 to 15. Fitting the visualizer to the dataset to determine the optimal number of clusters using the elbow method, and displaying the resulting visualization.

# In[19]:


from yellowbrick.cluster import KElbowVisualizer

model = KMeans(random_state = 1)
visualizer = KElbowVisualizer(model, k = (1,15))

visualizer.fit(dataset)
visualizer.show()
plt.show()


# 
# Initializing and fitting a KMeans model with 6 clusters, k-means++ initialization, and a random state of 0, then assigning the cluster labels to the variable y_kmeans

# In[20]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(dataset)


# In[21]:


y_kmeans


# Reshaping the y_kmeans array to have a single column using the reshape method and then concatenating it with the original dataframe along the second axis to create a new array bx.

# In[22]:


y_kmeans = y_kmeans.reshape(len(y_kmeans), 1)


# In[23]:


bx = np.concatenate((y_kmeans, dataframe), axis = 1)


# In[24]:


dataframe.columns


# In[25]:


dataframe_final = pd.DataFrame(data = bx, columns = ['Cluster_numbers','CustomerID', 'Gender', 'Age', 'Annual Income (k$)',
       'Spending Score (1-100)'])


# In[26]:


dataframe_final.head()


# In[27]:


dataframe_final.to_csv('clustered dataset')


# In[ ]:




