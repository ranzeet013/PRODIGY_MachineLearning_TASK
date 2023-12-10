#!/usr/bin/env python
# coding: utf-8

# ## Regression Analysis on House Price Dataset :

# This project focuses on conducting a comprehensive regression analysis on a house price dataset to predict housing prices based on various features. The dataset is sourced from Kaggle and contains information such as the number of bedrooms, square footage, location, and other relevant features.

# ### Importing Libraries :

# Importing Python libraries for data analysis and visualization, including pandas for data manipulation, numpy for numerical operations, seaborn and matplotlib for plotting, and yellowbrick for visualization of machine learning results, scikit-learn for machine learning tasks such as imputation and linear regression, and statsmodels for statistical modeling, including generalized linear models. 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm

from scipy import stats


from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error


# Importing train dataframe 

# In[2]:


train_df = pd.read_csv('train.csv', index_col = [0])


# In[3]:


train_df.head()


# In[4]:


train_df.shape


# Importing test dataframe 

# In[5]:


test_df = pd.read_csv('test.csv', index_col = [0])
test_df.head()


# In[6]:


test_df.shape


# ### Preprocessing :

# Preprocessing in the context of data analysis and machine learning involves the preparation and transformation of raw data to make it suitable for model training and analysis. It includes various steps such as handling missing values, scaling or normalizing features, encoding categorical variables, and splitting the data into training and testing sets. Preprocessing ensures that the data meets the requirements of the chosen machine learning algorithm, enhances model performance, and facilitates accurate and meaningful insights during analysis

# Concatenating two Pandas DataFrames, 'train_df' and 'test_df', vertically (along rows), and displaying the first few rows using the head().

# In[7]:


dataframe = pd.concat([train_df, test_df])
dataframe.head()


# In[8]:


dataframe.shape


# In[9]:


dataframe.info()


# In[10]:


dataframe.select_dtypes(include = 'object').head()


# In[11]:


dataframe.isna().sum()


# 
# Calculating the total number of missing values for each column in the 'dataframe' DataFrame using the isnull() method, summing them up, and subsequently sorting the results in descending order

# In[12]:


total_missing = dataframe.isnull().sum().sort_values(ascending = False)


# In[13]:


total_missing


# Computing the percentage of missing values for each column in the 'dataframe' DataFrame by dividing the count of missing values by the total count of values, and then sorting the results in descending order

# In[14]:


missing_in_percent = (dataframe.isnull().sum() / dataframe.isnull().count()).sort_values(ascending = False)


# In[15]:


missing_in_percent


# Combining the 'total_missing' and 'missing_in_percent' DataFrames horizontally (along columns) using the pd.concat function with specified column keys ("Total" and "Percent)

# In[16]:


missing_data = pd.concat([total_missing, missing_in_percent], 
                         axis = 1, 
                         keys = ["Total", "Percent"])


# In[17]:


missing_data.head(25)


# 
# Identifying columns with missing values exceeding 15% (excluding the "SalePrice" column), and then removing those columns, results in an updated 'dataframe'.

# In[18]:


missing_values = [column for column in dataframe.columns
                  if missing_in_percent.get(column, 0) > 0.15 and column != "SalePrice"]

dataframe = dataframe.loc[:, ~dataframe.columns.isin(missing_values)]
dataframe.info()


# ### SimpleImputer :

# SimpleImputer class from scikit-learn is utilized for handling missing values in a dataset. It offers a straightforward approach to replace missing values by employing different strategies such as mean, median, most frequent, or a constant value. By fitting the imputer to the dataset, it learns the required imputation strategy and can transform the data accordingly. 

# Creating a SimpleImputer object with the strategy set to "most_frequent" and applying it to fill missing values in the 'dataframe'

# In[19]:


imputer = SimpleImputer(strategy="most_frequent")

imputer.fit_transform(dataframe)

train_df.info()


# 
# Iterating through columns in the 'dataframe', checking if the data type is 'object', and converting those columns to numerical codes

# In[20]:


for feature in dataframe.columns:
    if dataframe[feature].dtype == 'object':
       dataframe[feature] = pd.Categorical(dataframe[feature]).codes


# Slicing the 'dataframe' to create a training DataFrame 'train_df' containing the first 1460 rows, and then defining predictor variables 'x' by dropping the "SalePrice" column and the target variable 'y' as 'train_df.SalePrice'.

# In[21]:


train_df = dataframe.iloc[:1460]

x = train_df.drop(["SalePrice"], axis = 1)
y = train_df.SalePrice


# In[22]:


x.head()


# In[23]:


y.head()


# ### Splitting Dataset : 

# 
# Splitting a dataset refers to dividing it into distinct subsets for training and testing purposes in machine learning. The typical split involves creating a training set, used to train and build a predictive model, and a testing set, employed to evaluate the model's performance on unseen data

# 
# Splitting the data into training and testing sets using the train_test_split function with a test size of 0.2 and a random state of 42. The resulting sets are 'x_train', 'x_test' for predictor variables, and 'y_train', 'y_test' for the target variable. 

# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# Calculating the first quartile (Q1) and the third quartile (Q3) for the predictor variables in the training set 'x_train' using the quantile method with the arguments 0.25 and 0.75,

# In[25]:


Q1 = x_train.quantile(0.25)

Q3 = x_train.quantile(0.75)


# In[26]:


Q1.head(25)


# In[27]:


Q3.head(25)


# ### The Interquartile Range (IQR) :

# The Interquartile Range (IQR) is a measure of statistical dispersion, representing the range between the first quartile (25th percentile) and the third quartile (75th percentile) of a dataset. It provides insights into the spread of the middle 50% of the data and is calculated as follows:
# 
# \[ IQR = Q3 - Q1 \]
# 
# where \( Q1 \) is the first quartile and \( Q3 \) is the third quartile. IQR is often used in statistics to identify and handle outliers, as values outside a certain range beyond the quartiles may be considered as potential outliers.

# 
# Calculating the Interquartile Range (IQR) by subtracting the first quartile (Q1) from the third quartile (Q3) for the predictor variables in the training set 'x_train'.

# In[28]:


IQR = Q3 - Q1

IQR.head(25)


# Calculating lower and upper bounds for outlier detection based on the first quartile (Q1), third quartile (Q3), and the Interquartile Range (IQR). The lower bound is computed as Q1−1.5×IQR and the upper bound as Q3+1.5×IQR. 

# In[29]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# ### Outlier :

#  An outlier is an observation that significantly deviates from the overall pattern of a dataset. In the context of linear regression or statistical analysis, outliers can exert a notable impact on the model's performance, potentially influencing parameter estimates and overall model fit. Identifying and handling outliers is crucial for ensuring the robustness and accuracy of statistical analyses, as these extreme observations can skew results and affect the reliability of predictions. 

# 
# Filtering out outliers from the training and testing sets based on the calculated lower and upper bounds.

# In[30]:


x_train_no_outlier = x_train[((x_train >= lower_bound) & (x_train <= upper_bound)).all(axis = 1)]
y_train_no_outlier = y_train.loc[x_train_no_outlier.index]

x_test_no_outlier = x_test[((x_test >= lower_bound) & (x_test <= upper_bound)).all(axis = 1)]
y_test_no_outlier = y_test.loc[x_test_no_outlier.index]


# ###  Linear Regression :

# Linear Regression is a statistical method used for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. In a simple linear regression, there is one independent variable, while multiple independent variables are considered in multiple linear regression. The model assumes a linear association between the variables, and the goal is to find the coefficients that minimize the sum of squared differences between the predicted and actual values. The resulting linear equation can be used for predicting the dependent variable based on the values of the independent variables. 

# 
# Creating a Linear Regression model ('lin_reg'), fitting it to the training data without outliers ('x_train_no_outlier', 'y_train_no_outlier'), and predicting the target variable for the training set

# In[31]:


lin_reg = LinearRegression()

lin_reg.fit(x_train_no_outlier, y_train_no_outlier)

y_pred = lin_reg.predict(x_train_no_outlier)


# 
# Displaying the coefficients (weights) and the intercept of the trained Linear Regression model.

# In[32]:


weights = lin_reg.coef_
intercept = lin_reg.intercept_

print('Coefficients: \n', weights[:25])
print('Interceptor: \n', intercept)


# ### Ordinary Least Squares (OLS) :

# Ordinary Least Squares (OLS) is a method used in linear regression analysis to estimate the parameters of a linear model. The goal of OLS is to find the line that minimizes the sum of the squared differences between the observed and predicted values of the dependent variable. In the context of linear regression, OLS aims to identify the coefficients for the predictor variables that best fit the observed data. The resulting model represents the linear relationship between the predictors and the dependent variable. 

# Performing Ordinary Least Squares (OLS) regression using the statsmodels library. The model is fitted to the training data without outliers.

# In[33]:


model = sm.OLS(y_train_no_outlier, sm.add_constant(x_train_no_outlier))

result = model.fit()

result.summary()


# Creating a probability plot (Q-Q plot) to visualize the residuals of the OLS regression model.

# In[34]:


fig, ax = plt.subplots(figsize = (8,4))
stats.probplot(result.resid,plot = plt)


# ### Quantile-Quantile (Q-Q) plot :

# A Quantile-Quantile (Q-Q) plot is a graphical tool used to assess whether a given dataset follows a specific theoretical distribution, such as a normal distribution.In the context of regression analysis, a Q-Q plot of residuals is often employed to check if the residuals conform to the assumptions of a normally distributed error term. If the residuals closely align with a straight line in the Q-Q plot, it suggests that the assumption of normality is reasonable. Deviations from the line may indicate departures from normality, prompting further investigation into the model's assumptions.

# Creating a residuals plot using the Yellowbrick library. The plot visualizes the residuals of the linear regression model on both the training and testing sets. The residuals_plot function is used with specified parameters, including the linear regression model, training data, and testing data. The plot includes a Q-Q plot and does not display a histogram. 

# In[35]:


plt.figure(figsize=(10, 5));
viz = residuals_plot(lin_reg, 
                     x_train_no_outlier, 
                     y_train_no_outlier, 
                     x_test_no_outlier, 
                     y_test_no_outlier, 
                     is_fitted = True, qqplot = True, hist = False)


#  The plot visualizes the difference between predicted and actual values on the testing set for the linear regression model. The prediction_error function is used with specified parameters, including the linear regression model, testing data.

# In[36]:


plt.figure(figsize = (8,4))
visualizer = prediction_error(lin_reg, 
                              x_test_no_outlier, 
                              y_test_no_outlier, 
                              is_fitted = True)


# ### MSE and R2 Score :

# Mean Squared Error (MSE) serves as a measure of the average squared differences between the predicted and actual values, providing insight into the model's accuracy. A lower MSE indicates a closer fit of the model to the data.
# 
# 
# R2 score, or coefficient of determination, quantifies the proportion of variance in the dependent variable explained by the model. This metric ranges from 0 to 1, where higher values denote a better ability of the model to capture the variability in the data. An R2 score of 1 signifies perfect prediction, while a score of 0 suggests the model lacks explanatory power.

# 
# Generating predictions using the trained OLS regression model ('result') on the training data without outliers. Calculating the mean squared error (MSE) between the predicted values and the actual target values using the mean_squared_error function

# In[37]:


prediction = result.predict(sm.add_constant(x_train_no_outlier))

mean_square_error = mean_squared_error(y_train_no_outlier, prediction)


# In[38]:


prediction


# In[39]:


mean_square_error


# Calculating the R2 (coefficient of determination) score between the predicted values and the actual target values on the training data without outliers. 

# In[40]:


R_squared = r2_score(y_train_no_outlier ,prediction)
R_squared


# In[ ]:




