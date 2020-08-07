#!/usr/bin/env python
# coding: utf-8

# In[1]:


# all the libraries needed

import numpy as np
import pandas as pd
#import tensorflow as tf
#from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

# for classification algorithm testing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, RobustScaler


# In[2]:


# original dataset

data = pd.read_csv("/Users/maggiesweeney/Downloads/CS_380/creditcard.csv", engine='python')

data.head()


# In[3]:



data.loc[:, ['Amount']].describe()


# In[4]:


# normalizing the data: need to scale Time and Amount to normalize it (all other columns are scaled due to PCA)

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['Amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['Time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

scaled_amount = data['Amount']
scaled_time = data['Time']

data.drop(['Amount', 'Time'], axis=1, inplace=True)
data.insert(0, 'Amount', scaled_amount)
data.insert(1, 'Time', scaled_time)

#data is normalized

data.head()


# In[5]:


# describing contents of the data

data.describe()


# In[6]:


# shows the classification distribution

print("0 = not fraud")
print("1 = fraud")
print((data.Class.value_counts(normalize = True) * 100))


# In[7]:


# chart of classification distribution

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data = data, palette = colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)


# In[ ]:





# In[8]:


# heatmap 

corr = data.corr()
plt.figure(figsize=(12,10))
heat = sns.heatmap(data=corr)
plt.title('Heatmap of Correlation')


# In[9]:


# need to split the data before we undersample so we test on the original dataset

X = data.drop('Class', axis=1)
y = data['Class']

splt = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# splits the data into 5 training & testing sets
# you should test on multiple splits of data 

for train_index, test_index in splt.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


# In[10]:


# randomly undersample the data to get even distribution of 'fraud' and 'not fraud'

data = data.sample(frac = 1)

# set the amount of nonfraud transactions to equal the amount of fraud transactions

fraud_data = data.loc[data['Class'] == 1]
nonfraud_data = data.loc[data['Class'] == 0][:492]


distributed_data = pd.concat([fraud_data, nonfraud_data])

# shuffle the new dataset

new_data = distributed_data.sample(frac=1, random_state=42)

# shows the evenly distributed data with selected columns

new_data[['Amount', 'Time', 'Class']]


# In[11]:


# shows the new distribution and chart

print("0 = not fraud")
print("1 = fraud")
print((new_data.Class.value_counts(normalize = True) * 100))

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data = new_data, palette = colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)


# In[12]:


# heatmap 

new_corr = new_data.corr()
plt.figure(figsize=(12,10))
heat = sns.heatmap(new_corr)
plt.title('Heatmap of Correlation')


# In[13]:


# show high positive correlations
corr[new_corr.Class > 0.5]


# In[14]:


# show low negative correlations
corr[new_corr.Class < -0.5]


# In[113]:


# Only removing extreme outliers
Q1 = new_data.quantile(0.25)
Q3 = new_data.quantile(0.75)

# Finding the interquartile range (difference between upper quartile (highest 25%) & lower quartile (lowest 25%))
IQR = Q3 - Q1

# Need to understnad this equation
fresh_data = new_data[~((new_data < (Q1 - 1.5 * IQR)) |(new_data > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[114]:


# Prints how many outliers were deleted

len_after = len(fresh_data)
len_before = len(new_data)
len_difference = len(new_data) - len(fresh_data)
print('We reduced our data size from {} transactions by {} transactions to {} transactions.'.format(len_before, len_difference, len_after))


# In[115]:


# Reducing dimensionality

X = fresh_data.drop('Class', axis=1)
y = fresh_data['Class']

#t-SNE

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)


# In[116]:


# split training and testing data

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_validation = X_test.values
y_train = y_train.values
y_validation = y_test.values


# In[117]:


# models = []

# models.append(('Logisitc Regression', LogisticRegression()))
# models.append(('Linear Regression', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('Decision Tree', DecisionTreeClassifier()))
# models.append(('Support Vector Machine', SVC()))
# models.append(('Random Forest', RandomForestClassifier()))

#testing models

# results = []
# names = []

# for name, model in models:
#    kfold = KFold(n_splits=10, random_state=42)
#    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
#    results.append(cv_results)
#    names.append(name)
#    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
#    print(msg)


# In[118]:


# Logistic Regression - training the data


logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)


# In[119]:


# Logistic regreaaion - testing the data

pred = logistic_regression.predict(X_test)


# In[120]:


# Testing the accuracy

train_accuracy_percentage = metrics.accuracy_score(y_train,logistic_regression.predict(X_train))
train_accuracy_percentage = train_accuracy_percentage * 100

print('The train set accuracy is {} %'.format(train_accuracy_percentage))

accuracy = metrics.accuracy_score(y_test, pred)
test_accuracy_percentage = 100 * accuracy
print('The test set accuracy is {} %'.format(test_accuracy_percentage))


# In[121]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[122]:


# Confusion matrix

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 15);


# In[ ]:




