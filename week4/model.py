#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd


# In[47]:


# Load data sets 
iris = load_iris()
iris


# In[48]:


X, y = iris.data, iris.target


# In[90]:


# Data preprocessing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)



# In[73]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[81]:


#random forest model (rf) (initial version)
from sklearn.ensemble import RandomForestClassifier

#random forest model
clf_rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)

#produce true vs predicted label 
predictions_rf = clf_rf.predict(X_test)
#produce classification matrix
print(classification_report(y_test,predictions_rf))
#confusion matrix
pd.crosstab(y_test, predictions_rf, rownames=["Actual"], colnames=["Predicted"])


# In[75]:


# Check for overfitting
clf_rf = RandomForestClassifier(max_depth=2, random_state=0)

# Perform cross-validation
train_scores = cross_val_score(clf_rf, X_train, y_train, cv=5, scoring='accuracy')
val_scores = cross_val_score(clf_rf, X_test, y_test, cv=5, scoring='accuracy')

# Print the training and cross-validation accuracies
print("Training Accuracy:", train_scores.mean())
print("Cross-Validation Accuracy:", val_scores.mean())


# In[93]:


import pickle

clf_rf.fit(X_train, y_train)

pickle.dump(clf_rf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(clf_rf.predict([[3.5, 1.2, 2.4,0.2]]))


# In[ ]:





# In[ ]:





# In[ ]:




