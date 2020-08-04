#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection
# 
# In this project you will predict fraudulent credit card transactions with the help of Machine learning models. Please import the following libraries to get started.

# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from scipy import interp
import itertools


# ## Exploratory data analysis

# In[5]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[6]:


#Missing Value Analysis
df.isnull().values.any()


# In[7]:


#observe the different feature type present in the data
df.info()


# In[8]:


df.describe() #Describing the distribution of the each column


# Here we will observe the distribution of our classes

# In[9]:


classes=df['Class'].value_counts()
normal_share=classes[0]/df['Class'].count()*100
fraud_share=classes[1]/df['Class'].count()*100


# # EXPLORATORY DATA ANALYSIS

# ###### Percentage of regular and fraud Transactions

# In[10]:


print('Percentage of Normal Transactions: {}%'.format(round(df.Class.value_counts()[0]/len(df) * 100.0,2)))
print('Percentage of Frauds: {}%'.format(round(df.Class.value_counts()[1]/len(df) * 100.0,2)))


# ###### Normal and fraud Dataset Creation

# In[11]:


fraud = df[df['Class']==1]
normal = df[df['Class']==0]
fraud.head()


# In[12]:


# Create a bar plot for the number and percentage of fraudulent vs non-fraudulent transcations
#sns.barplot(x= 'Normal' y= 'Fraud', data = df)
LABELS = ["Normal", "Fraud"]
count_classes = pd.value_counts(df['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[13]:


# Create a scatter plot to observe the distribution of classes with time
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time vs Amount divided by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time given in Seconds')
plt.ylabel('Amount')
plt.show()


# In[14]:


# Create a scatter plot to observe the distribution of classes with Amount
plt.scatter(df.Class, df.Amount)
plt.xlabel('Class')
plt.ylabel('Amount')
plt.show()


# In[15]:


# Let us further analyze the variable "Amount"
plt.boxplot(df['Amount'], labels = ['Boxplot'])
plt.ylabel('Transaction amount')
plt.plot()

amount = df[['Amount']].sort_values(by='Amount')


# In[16]:


q1, q3 = np.percentile(amount,[25,75])
q1,q3


# In[17]:


iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)
print('# outliers below the lower bound: ', amount[amount['Amount'] < lower_bound].count()[0],
     ' ({:.4}%)'.format(amount[amount['Amount'] < lower_bound].count()[0] / amount['Amount'].count() * 100))
print('# outliers above the upper bound: ', amount[amount['Amount'] > upper_bound].count()[0],
      ' ({:.4}%)'.format(amount[amount['Amount'] > upper_bound].count()[0] / amount['Amount'].count() * 100))


# If we delete this outliers we will be losing about 11.2% of Data

# In[18]:


#Checking Corelation with the Heatmap
heatmap = sns.heatmap(df.corr(method='spearman'),cmap='coolwarm',robust = True)


# In[19]:


heatmap = sns.heatmap(df.corr(method='pearson'),cmap='coolwarm',robust = True)


# Because of PCA Transformation we have not found any Corelation Issues

# In[20]:


# Drop unnecessary columns
df = df.drop(['Time'], axis=1)


# Since Time Variable will not contribute much to our model we are dropping it for better efficiency

# In[21]:


df.shape


# ##### DATA SCALING

# In[22]:


from sklearn.preprocessing import StandardScaler
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df.head()


# We can see the situation of class imbalance clearly

# ### Splitting the data into train & test data

# In[23]:


y= df['Class']
X = df.drop(columns=['Class'])


# In[24]:


X.head()


# In[25]:


y.head()


# In[26]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold


# In[27]:


from sklearn import model_selection

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, random_state=42)


# ##### Preserve X_test & y_test to evaluate on the test data once you build the model

# ### MODEL BUILDING

# In[28]:


import statsmodels.api as sm


# In[29]:


# Building simple Logistic regression model without balancung the result
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[30]:


#Feature selection using RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[31]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # we are running the RFE as 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[32]:


rfe.support_


# In[33]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[34]:


col = X_train.columns[rfe.support_]


# In[35]:


X_train.columns[~rfe.support_]


# In[36]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[37]:


#Acess the model with the stats madel
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]


# In[38]:


y_train_pred_final = pd.DataFrame({'Class':y_train.values, 'Class_Prob':y_train_pred})
y_train_pred_final.head()


# In[39]:


y_train_pred_final['predicted'] = y_train_pred_final.Class_Prob.map(lambda x: 0 if x < 0.01 else 1)
y_train_pred_final.tail()


# In[40]:


from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final.predicted )
print(confusion)


# In[41]:


#Checking the Accuracy
print(metrics.accuracy_score(y_train_pred_final.Class, y_train_pred_final.predicted))


# ## Model building with balancing Classes
# 
# ##### Perform class balancing with :
# - Random Oversampling
# - SMOTE
# - ADASYN
# - Random Undersampling

# In[42]:


X_train.head()


# ## Random Undersampling

# In this method, you have the choice of selecting fewer data points from the majority class for your model-building process. In case you have only 500 data points in the minority class, you will also have to take 500 data points from the majority class; this will make the classes somewhat balanced. However, in practice, this method is not effective because you will lose over 99% of the original data.

# In[43]:


rus = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)
X_rus, y_rus = rus.fit_resample(X_train, y_train)


# In[44]:


X_rus.shape,y_rus.shape


# In[45]:


y_rus.head()


# In[46]:


plt.bar(['Non-Fraud','Fraud'], [Counter(y_rus)[0], Counter(y_rus)[1]], color=['b','r'])
plt.xlabel('Class')
plt.ylabel('# transactions')

plt.tight_layout()
plt.show()


# In[47]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_rus)))


# ## Random Oversampling

# Random oversampling involves randomly minority points from the minority to group to match the length of the majority class. The process is entirely randowm it takes few rows from the minority class and adds up

# In[48]:


ros = RandomOverSampler(sampling_strategy='auto', random_state=48)
X_ros, y_ros = ros.fit_resample(X_train, y_train)


# In[49]:


plt.bar(['Non-Fraud','Fraud'], [Counter(y_ros)[0], Counter(y_ros)[1]], color=['b','r'])
plt.xlabel('Class')
plt.ylabel('# transactions')

plt.tight_layout()
plt.show()


# In[50]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_ros)))


# ## Synthetic Minority Over-Sampling Technique (SMOTE)

# In this process, you can generate new data points, which lie vectorially between two data points that belong to the minority class. These data points are randomly chosen and then assigned to the minority class. This method uses K-nearest neighbours to create random synthetic samples

# In[51]:


smote = SMOTE(sampling_strategy='auto', random_state=48)
X_smote, y_smote = smote.fit_resample(X_train, y_train)


# In[52]:


plt.bar(['Non-Fraud','Fraud'], [Counter(y_smote)[0], Counter(y_smote)[1]], color=['b','r'])
plt.xlabel('Class')
plt.ylabel('# transactions')

plt.tight_layout()
plt.show()


# In[53]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_smote)))


# ## ADASyn(Adaptive synthesis)

# This is similar to SMOTE, with a minor change in the generation of synthetic sample points for minority data points. For a particular data point, the number of synthetic samples that it will add will have a density distribution, whereas, for SMOTE, the distribution will be uniform. The aim here is to create synthetic data for minority examples that are harder to learn, rather than the easier ones

# In[54]:


from imblearn.over_sampling import ADASYN


# In[55]:


ads = ADASYN(sampling_strategy='auto', random_state=48)
X_ads, y_ads = ads.fit_resample(X_train, y_train)


# In[56]:


plt.bar(['Non-Fraud','Fraud'], [Counter(y_ads)[0], Counter(y_ads)[1]], color=['b','r'])
plt.xlabel('Class')
plt.ylabel('# transactions')

plt.tight_layout()
plt.show()


# In[57]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_ads)))


# We will create 2d plot to visualize the transformed data

# In[58]:


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#8c564b', '#FF7F0E']
    markers = ['v', '^']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


# In[59]:


pca = PCA(n_components=2)
X_ros_pca = pca.fit_transform(X_ros)
X_smote_pca = pca.fit_transform(X_smote)
X_ads_pca = pca.fit_transform(X_ads)
X_rus_pca = pca.fit_transform(X_rus)


# In[60]:


plot_2d_space(X_ros_pca, y_ros, 'Balanced dataset PCA_transformed using random oversampling')
plot_2d_space(X_smote_pca, y_smote, 'Balanced dataset PCA_transformed using SMOTE')


# In[61]:


plot_2d_space(X_ads_pca, y_ads, 'Balanced dataset PCA_transformed using adaptive synthesis')
plot_2d_space(X_rus_pca, y_rus, 'Balanced dataset PCA_transformed using random undersampling')


# ## Logistic regression on random undersampling data

# In[62]:


lr = LogisticRegression(max_iter=200, random_state=0, n_jobs = -1)
lr.fit(X_rus, y_rus)


# In[63]:


lr_predict = lr.predict(X_test)


# In[64]:


lr_predict


# ## Prediction scores

# In[65]:


from sklearn.metrics import accuracy_score, recall_score, confusion_matrix,roc_auc_score
from matplotlib import pyplot

lr_accuracy = accuracy_score(y_test, lr_predict)
lr_recall = recall_score(y_test, lr_predict)
lr_cm = confusion_matrix(y_test, lr_predict)
lr_auc = roc_auc_score(y_test, lr_predict)

print("Accuracy: {:.4%}".format(lr_accuracy))
print("Recall: {:.4%}".format(lr_recall))
print("ROC AUC: {:.4%}".format(lr_auc))

lr_cm = pd.DataFrame(lr_cm, ['normal','fraud'],['Predicted_Normal','Predicted_Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(lr_cm, annot=True,annot_kws={"size": 16},fmt='g')


# ## Logistic regression on random oversampling data¶

# In[66]:


lr = LogisticRegression(max_iter=200, random_state=0, n_jobs = -1)
lr.fit(X_ros, y_ros)


# In[67]:


lr_predict_ros = lr.predict(X_test)


# In[68]:


lr_accuracy_ros = accuracy_score(y_test, lr_predict_ros)
lr_recall_ros = recall_score(y_test, lr_predict_ros)
lr_cm_ros = confusion_matrix(y_test, lr_predict_ros)
lr_auc_ros = roc_auc_score(y_test, lr_predict_ros)

print("Accuracy: {:.4%}".format(lr_accuracy_ros))
print("Recall: {:.4%}".format(lr_recall_ros))
print("ROC AUC: {:.4%}".format(lr_auc_ros))

lr_cm = pd.DataFrame(lr_cm, ['normal','fraud'],['Predicted_Normal','Predicted_Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(lr_cm, annot=True,annot_kws={"size": 16},fmt='g')


# ## Logistic regression on SMOTE oversampling data

# In[69]:


lr = LogisticRegression(max_iter=200, random_state=0, n_jobs = -1)
lr.fit(X_smote, y_smote)


# In[70]:


lr_predict_smote = lr.predict(X_test)


# In[71]:


lr_accuracy_smote = accuracy_score(y_test, lr_predict_smote)
lr_recall_smote = recall_score(y_test, lr_predict_smote)
lr_cm_smote = confusion_matrix(y_test, lr_predict_smote)
lr_auc_smote = roc_auc_score(y_test, lr_predict_smote)

print("Accuracy: {:.4%}".format(lr_accuracy_smote))
print("Recall: {:.4%}".format(lr_recall_smote))
print("ROC AUC: {:.4%}".format(lr_auc_smote))

lr_cm = pd.DataFrame(lr_cm, ['normal','fraud'],['Predicted_Normal','Predicted_Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(lr_cm, annot=True,annot_kws={"size": 16},fmt='g')


# ## Logistic regression on Adasyn oversampling data

# In[72]:


lr = LogisticRegression(max_iter=200, random_state=0, n_jobs = -1)
lr.fit(X_ads, y_ads)


# In[73]:


lr_predict_ads = lr.predict(X_test)


# In[74]:


lr_accuracy_ads = accuracy_score(y_test, lr_predict_ads)
lr_recall_ads = recall_score(y_test, lr_predict_ads)
lr_cm_ads = confusion_matrix(y_test, lr_predict_ads)
lr_auc_ads = roc_auc_score(y_test, lr_predict_ads)

print("Accuracy: {:.4%}".format(lr_accuracy_ads))
print("Recall: {:.4%}".format(lr_recall_ads))
print("ROC AUC: {:.4%}".format(lr_auc_ads))

lr_cm = pd.DataFrame(lr_cm, ['normal','fraud'],['Predicted_Normal','Predicted_Fraud'])
pyplot.figure(figsize = (8,4))
sns.set(font_scale=1.4)
sns.heatmap(lr_cm, annot=True,annot_kws={"size": 16},fmt='g')


# perfom cross validation on the X_train & y_train to create:
# X_train_cv
# X_test_cv
# y_train_cv
# y_test_cv

# ###### Similarly explore other algorithms on balanced dataset by building models like:

# ###### KNN Random Forest XGBoost

# Apart from logistic regression let us explore other option since it a classification problem logistic regression is prefferred over all other

# In[75]:


from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp


# Using SMOTE we got: Accuracy: 97.6780% Recall: 87.8378% ROC AUC: 92.7664% Also we are not loosing any information hence we will use this technique further

# # Random Forest on SMOTE

# Let's first fit a random forest model with default hyperparameters.

# one of the most popular algorithms in machine learning. Random forests use a technique known as bagging, which is an ensemble method. So before diving into random forests, let's first understand ensembles.

# In[76]:



# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[77]:


# fit
rfc.fit(X_smote,y_smote)


# In[78]:


# Making predictions
predictions = rfc.predict(X_test)


# In[79]:


# Making predictions
predictions = rfc.predict(X_test)


# In[80]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[81]:


# Printing confusion matrix
print(confusion_matrix(y_test,predictions))


# In[82]:


print(accuracy_score(y_test,predictions))


# let's now look at the list of hyperparameters which we can tune to improve model performance.

# In[83]:


model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')


# In[84]:


# Fit on training data
model.fit(X_smote, y_smote)


# In[85]:


# Making predictions
predictions = model.predict(X_test)


# In[86]:


print(classification_report(y_test,predictions))


# In[87]:


# Probabilities for each class
rf_probs = model.predict_proba(X_test)[:, 1]


# In[88]:


from sklearn.metrics import roc_auc_score

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)


# In[89]:


roc_value


# In[90]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[91]:


fpr, tpr, thresholds = metrics.roc_curve( y_test, rf_probs, drop_intermediate = False )

draw_roc(y_test, rf_probs)


# # Random Forest on random oversampling

# In[92]:


# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[93]:


# fit
rfc.fit(X_ros, y_ros)


# In[94]:


# Making predictions
predictions = rfc.predict(X_test)


# In[95]:


# Making predictions
predictions = rfc.predict(X_test)


# In[96]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[97]:


# Printing confusion matrix
print(confusion_matrix(y_test,predictions))


# In[99]:


print(accuracy_score(y_test,predictions))


# let's now look at the list of hyperparameters which we can tune to improve model performance.

# In[100]:


model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')


# In[101]:


# Fit on training data
model.fit(X_ros, y_ros)


# In[102]:


# Making predictions
predictions = rfc.predict(X_test)


# In[103]:


print(classification_report(y_test,predictions))


# In[104]:


# Probabilities for each class
rf_probs = model.predict_proba(X_test)[:, 1]


# In[105]:


from sklearn.metrics import roc_auc_score

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)


# In[ ]:


roc_value


# In[106]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[107]:


fpr, tpr, thresholds = metrics.roc_curve( y_test, rf_probs, drop_intermediate = False )

draw_roc(y_test, rf_probs)


# # Random Forest on ADS

# In[111]:


# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[112]:


# fit
rfc.fit(X_ads, y_ads)


# In[113]:


# Making predictions
predictions = rfc.predict(X_test)


# In[114]:


# Making predictions
predictions = rfc.predict(X_test)


# In[115]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# In[116]:


# Printing confusion matrix
print(confusion_matrix(y_test,predictions))


# In[117]:


print(accuracy_score(y_test,predictions))


# let's now look at the list of hyperparameters which we can tune to improve model performance.

# In[118]:


model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')


# In[119]:


# Fit on training data
model.fit(X_ads, y_ads)


# In[120]:


# Making predictions
predictions = rfc.predict(X_test)


# In[121]:


print(classification_report(y_test,predictions))


# In[122]:


# Probabilities for each class
rf_probs = model.predict_proba(X_test)[:, 1]


# In[123]:


from sklearn.metrics import roc_auc_score

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)


# In[124]:


roc_value


# In[125]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[126]:


fpr, tpr, thresholds = metrics.roc_curve( y_test, rf_probs, drop_intermediate = False )

draw_roc(y_test, rf_probs)


# ## XG BOOST ON SMOTE

# In[127]:


import xgboost as xgb


# In[128]:


from xgboost import XGBClassifier
tree_range = range(2, 30, 5)
score1=[]
score2=[]
for tree in tree_range:
    xgb=XGBClassifier(n_estimators=tree)
    xgb.fit(X_smote,y_smote)
    score1.append(xgb.score(X_smote,y_smote))
    score2.append(xgb.score(X_test,y_test))
    
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(tree_range,score1,label= 'Accuracy on training set')
plt.plot(tree_range,score2,label= 'Accuracy on testing set')
plt.xlabel('Value of number of trees in XGboost')
plt.ylabel('Accuracy')
plt.legend()


# As we can see accuracy is increasing for the test and stabilizes at one point

# In[129]:


xgb=XGBClassifier(n_estimators=18)
xgb.fit(X_smote,y_smote)
print('Accuracy of XGB on the testing dataset is :{:.3f}'.format(xgb.score(X_test,y_test)))


# In[130]:


# we got a 98% score using xgboost


# In[131]:


print(xgb.feature_importances_)


# In[132]:


pyplot.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
pyplot.show()


# In[133]:


from xgboost import plot_importance
plot_importance(xgb)
pyplot.show()


# # XG BOOST ON RANDOM OVERSAMPLING

# In[134]:


import xgboost as xgb


# In[135]:


from xgboost import XGBClassifier
tree_range = range(2, 30, 5)
score1=[]
score2=[]
for tree in tree_range:
    xgb=XGBClassifier(n_estimators=tree)
    xgb.fit(X_ros,y_ros)
    score1.append(xgb.score(X_ros,y_ros))
    score2.append(xgb.score(X_test,y_test))
    
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(tree_range,score1,label= 'Accuracy on training set')
plt.plot(tree_range,score2,label= 'Accuracy on testing set')
plt.xlabel('Value of number of trees in XGboost')
plt.ylabel('Accuracy')
plt.legend()


# In[136]:


xgb=XGBClassifier(n_estimators=18)
xgb.fit(X_ros,y_ros)
print('Accuracy of XGB on the testing dataset is :{:.3f}'.format(xgb.score(X_test,y_test)))


# In[137]:


print(xgb.feature_importances_)


# In[138]:


pyplot.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
pyplot.show()


# In[139]:


from xgboost import plot_importance
plot_importance(xgb)
pyplot.show()


# ## XG BOOST ON RANDOM ADS

# In[ ]:


import xgboost as xgb


# In[141]:


from xgboost import XGBClassifier
tree_range = range(2, 30, 5)
score1=[]
score2=[]
for tree in tree_range:
    xgb=XGBClassifier(n_estimators=tree)
    xgb.fit(X_ads,y_ads)
    score1.append(xgb.score(X_ads,y_ads))
    score2.append(xgb.score(X_test,y_test))
    
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(tree_range,score1,label= 'Accuracy on training set')
plt.plot(tree_range,score2,label= 'Accuracy on testing set')
plt.xlabel('Value of number of trees in XGboost')
plt.ylabel('Accuracy')
plt.legend()


# In[142]:


xgb=XGBClassifier(n_estimators=18)
xgb.fit(X_ads,y_ads)
print('Accuracy of XGB on the testing dataset is :{:.3f}'.format(xgb.score(X_test,y_test)))


# In[143]:


print(xgb.feature_importances_)


# In[144]:


pyplot.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
pyplot.show()


# In[145]:


from xgboost import plot_importance
plot_importance(xgb)
pyplot.show()


# #### 3. Cross-Validation:

# The following figure illustrates k-fold cross-validation with k=4. There are some other schemes to divide the training set, we'll look at them briefly later.

# ### K-Fold Cross Validation

# It is a statistical technique which enables us to make extremely efficient use of available data It divides the data into several pieces, or 'folds', and uses each piece as test data one at a time

# In[103]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[104]:


xgb=XGBClassifier(n_estimators=18)
scores = cross_val_score(xgb, X_smote, y_smote, scoring='r2', cv=5)
scores


# In[105]:


# the other way of doing the same thing (more explicit)

# create a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores_1 = cross_val_score(xgb, X_smote, y_smote, scoring='r2', cv=folds)
scores_1


# We used several methods to predict the default the best result we got by using XGboost on data which was sampled using SMOTE the Accuracy of XGB on the testing dataset is :0.981. Also the important features are:V4,V14,V12,V16,V11. Also by performing logistic regression we got a good score of Accuracy: 97.6780% Recall: 87.8378% ROC AUC: 92.7664% For classification model.

# In[ ]:




