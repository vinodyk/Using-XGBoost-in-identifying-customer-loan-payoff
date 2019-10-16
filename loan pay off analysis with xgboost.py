# -*- coding: utf-8 -*-
"""

## Data Science Challenge – The Development and Evaluation of a Supervised Model

# It should not be taken as exhaustive list of things to do with a dataset.
# It has been setup to provide Exploratory analysis, model development, and model performance.

#Author: Vinod Yadav
# Import file from url and un-zip to local drive for read
"""
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
geturl = 'https://yoururl.com/example.zip'
with urlopen(geturl) as zipdata:
    with ZipFile(BytesIO(zipdata.read())) as datafile:
        datafile.extractall('../temp')

import numpy as np # To process linear algebra
import pandas as pd # For data processing

# There are missing data fields that will be imputed
from fancyimpute import IterativeImputer as MICE
from sklearn.model_selection import cross_val_score

# Encoding the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Model selection
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# Metrics and performance accuracy of models
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc,confusion_matrix,recall_score
from xgboost import XGBClassifier as XGB
from xgboost import plot_importance

# Visualization
import matplotlib.pyplot as plt
import missingno
from IPython.display import display
import seaborn as sns
sns.set_style('whitegrid')

# Ignore depraction and other warnings
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Read what is in loan data
loans = pd.read_csv("../lacare/loan_data.csv")
loans.head()
#loans.info()

# Plot graphic of missing values
missingno.matrix(loans, figsize = (30,10))

# Count the number of NaN values in each column
print(loans.isnull().sum())

# Data type for emp_length is not valid, change it to str
loans.loan_status = loans.loan_status.replace({"Fully Paid": 0,"Charged Off": 1})
set(loans.loan_status)

# Replace n/a values with nulls for further imputation
loans.emp_length = loans.emp_length.replace({"n/a": np.NaN})
loans.emp_length = loans.emp_length.astype("str")

# Label encoding
columns = ["emp_length","addr_state", "home_ownership","application_type","verification_status","term","grade","purpose"]
for col in columns:
               loans[col] = LabelEncoder().fit_transform(loans[col])
# One hot encoding method applied to the categorical variables to convert  into a non-ordinal numerical representation.
loans = pd.get_dummies(loans, columns = columns, prefix = columns)
loans.head()

# Check for Data balance
fully_paid = len(loans[loans['loan_status'] == 0]['loan_status'])
charged_off = len(loans[loans['loan_status'] == 1]['loan_status'])
fully_paid_perc = round(fully_paid/len(loans)*100,1)
charged_off_perc = round(charged_off/len(loans)*100,1)

print('Number of clients that have Fully paid : {} ({}%)'.format(fully_paid, fully_paid_perc))
print('Number of clients that haven Charged off: {} ({}%)'.format(charged_off, charged_off_perc))

# Found that given data is imbalanced because clients that have "Fully paid": 98.0% vs "Charged off": 2.0%

# Use MICE Imputation method to fill missing rows
# MICE method uses Bayesian ridge regression avoids baises, fancyimpute removes column names
train_cols=list(loans)
loans=pd.DataFrame(MICE(verbose=False).fit_transform(loans))
loans.columns =train_cols

# Check data after imputation
loans.head()

# As the data is imbalaced and contain features in different order of magnitude Ensemble methods with random sampling work better.
# Will use XGB: Gradient Boosting Classifier. I could use Light GBM or CatBoost as well for real world comparison.

features = list(loans.drop('loan_status', axis = 1))
target = 'loan_status'

# Splitting the dataset into the Training set and Test set 80% , 20%
train, test = train_test_split(loans, test_size = 0.2, random_state = 1)

print('Number of clients in the train set: {}'.format(len(train)))
print('Number of clients in the test set: {}'.format(len(test)))

# The data contain features in different order of magnitude, some values are outliers which need normalization.
# Will normalize the data for a better performance. I am using StandardScaler normalization.
nrml = StandardScaler()

# Fit on training set
train[features] = nrml.fit_transform(train[features])

# Only transform on test set
test[features] = nrml.transform(test[features])

# Xtreme Gradient Boosting model, this is proven model for high accuracy and performance.
model_XGB = XGB(max_depth = 6,
            learning_rate = .1,
            n_estimators = 100,
            reg_lambda = 0.5,
            reg_alpha = 0,
            verbosity = 1,
            n_jobs = -1,
            tree_method = 'exact').fit(train[features], train[target])
            # not using GPU as my laptop hardware doesn't support

pred = model_XGB.predict(test[features])
predp = model_XGB.predict_proba(test[features])[:,1]

plot_importance(model_XGB,max_num_features=10) # top 10 most important features

plt.show()



fpr_cv, tpr_cv, thresholds_cv = roc_curve(pred, predp)
roc_auc_cv = auc(fpr_cv,tpr_cv)
print('auc:', roc_auc_cv)
# ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_cv, tpr_cv, 'b',label='AUC = %0.4f'% roc_auc_cv)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
