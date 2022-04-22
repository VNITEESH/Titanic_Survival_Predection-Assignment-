
from django import conf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
data=pd.read_csv('/Users/nithish/Desktop/titanic_dataset/train (1).csv')
print(data.head(2))

print(data.corr())
print(data.describe())
print(data.describe().T)

## Checking the null values ....
print(data.isna().sum())
data['Cabin'].fillna(data['Cabin'].ffill(),inplace=True)
data['Age'].fillna(data['Age'].mean(),inplace=True)
print(data.isna().sum())
data['Cabin'].fillna(data['Cabin'].bfill(),inplace=True)
print(data.isna().sum())
data['Embarked'].fillna(data['Embarked'].bfill(),inplace=True)
print(data.isna().sum())
print(data.dtypes)


## EDA..
"""
sns.pairplot(data)
plt.show()
"""
print(data.dtypes)
"""
plt.subplot(1,6,1)
plt.violinplot(data['Survived'])
plt.subplot(1,6,2)
plt.violinplot(data['Pclass'])
plt.subplot(1,6,3)
plt.violinplot(data['Age'])
plt.subplot(1,6,4)
plt.violinplot(data['SibSp'])
plt.subplot(1,6,5)
plt.violinplot(data['Parch'])
plt.subplot(1,6,6)
plt.violinplot(data['Fare'])

plt.title('Violin plot of the features')
plt.show()


plt.subplot(1,6,1)
sns.boxenplot(data['Survived'])
plt.subplot(1,6,2)
sns.boxenplot(data['Pclass'])
plt.subplot(1,6,3)
sns.boxenplot(data['Age'])
plt.subplot(1,6,4)
sns.boxenplot(data['SibSp'])
plt.subplot(1,6,5)
sns.boxenplot(data['Parch'])
plt.subplot(1,6,6)
sns.boxenplot(data['Fare'])

plt.title('boxen plot of the features')
plt.show()

plt.subplot(1,6,1)
sns.boxplot(data['Survived'])
plt.subplot(1,6,2)
sns.boxplot(data['Pclass'])
plt.subplot(1,6,3)
sns.boxplot(data['Age'])
plt.subplot(1,6,4)
sns.boxplot(data['SibSp'])
plt.subplot(1,6,5)
sns.boxplot(data['Parch'])
plt.subplot(1,6,6)
sns.boxplot(data['Fare'])

plt.title('boxen plot of the features')
plt.show()

plt.subplot(1,6,1)
sns.distplot(data['Survived'])
plt.subplot(1,6,2)
sns.distplot(data['Pclass'])
plt.subplot(1,6,3)
sns.distplot(data['Age'])
plt.subplot(1,6,4)
sns.distplot(data['SibSp'])
plt.subplot(1,6,5)
sns.distplot(data['Parch'])
plt.subplot(1,6,6)
sns.distplot(data['Fare'])

plt.title('bist plot of the features')
plt.show()


sns.heatmap(data.corr())
plt.show()

"""

data['Embarked']=data['Embarked'].astype('category')
print(data.dtypes)
## convert the string features into numeric...
print(data.dtypes)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Name']=le.fit_transform(data['Name'])
data['Sex']=le.fit_transform(data['Sex'])
data['Ticket']=le.fit_transform(data['Ticket'])
data['Embarked']=le.fit_transform(data['Embarked'])
data['Cabin']=le.fit_transform(data['Cabin'])
print(data.dtypes)



## Split the data..
x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print(x.head())
print(y.head())





## Training and testing....
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.3)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

## apply the algorithm...
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypred_lr=lr.predict(xtest)


### performance testing....
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('performance testing ')
print('===========================================')
print('confusion matrix:',confusion_matrix(ytest,ypred_lr))
print('===========================================')
print('accuracy score:',accuracy_score(ytest,ypred_lr))
print('===========================================')
print('classification report:',classification_report(ytest,ypred_lr))
print("======================================================")
print("THE NAIVE BAYES ALOGORITHM")
#### Select the model....
from sklearn.naive_bayes import GaussianNB
gaus=GaussianNB()
gaus.fit(xtrain,ytrain)
ypred_gaus=gaus.predict(xtest)
print('the test size of the test model is',len(ypred_gaus))

#### Performance testing ...
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("The performance testing for the GaussianNB")
print('=========================================================')
print("confusion matrix",confusion_matrix(ytest,ypred_gaus))
print('=========================================================')
print("accuracy score",accuracy_score(ytest,ypred_gaus))
print('=========================================================')
print('classification report:',classification_report(ytest,ypred_gaus))
print('=========================================================')

###  Multinomial navie bayes....
from sklearn.naive_bayes import MultinomialNB
multi=MultinomialNB()
multi.fit(xtrain,ytrain)
ypred_multi=multi.predict(xtest)
print('the test size is:',len(ypred_multi))

### performance testing for the multinomial
from sklearn.metrics import classification_report,accuracy_score,multilabel_confusion_matrix,confusion_matrix
print("The performace testing for the multinomial navie bayes")
print('=========================================================')
print('multilabel confusion matrix:',multilabel_confusion_matrix(ytest,ypred_multi))
print('=========================================================')
print('accuracy score :',accuracy_score(ytest,ypred_multi))
print('=========================================================')
print('classification report:',classification_report(ytest,ypred_multi))
print('=========================================================')


print(multi.class_count_)
print(multi.class_log_prior_)
print(multi.classes_)
print(multi.feature_log_prob_)



### 
from sklearn.naive_bayes import BernoulliNB
bb=BernoulliNB()
bb.fit(xtrain,ytrain)
ypred_bb=bb.predict(xtest)


### performance testing for bernouli...
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("performance testing for bernouli")
print('=========================================================')
print('confusion matrix :',confusion_matrix(ytest,ypred_bb))
print('=========================================================')
print('accuracy score:',accuracy_score(ytest,ypred_bb))
print('=========================================================')
print('classification report:',classification_report(ytest,ypred_bb))
print('=========================================================')

### Random Forest Classifier...
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=5)
rfc.fit(xtrain,ytrain)
ypred_rfc=rfc.predict(xtest)
print(ypred_rfc)

##Performance Testing For Random Forest Classifier
print("performance testing for Random Forest Classifier")
print('=========================================================')
print('confusion matrix :',confusion_matrix(ytest,ypred_rfc))
print('=========================================================')
print('accuracy score:',accuracy_score(ytest,ypred_rfc))
print('=========================================================')
print('classification report:',classification_report(ytest,ypred_rfc))
print('=========================================================')



### All Accuracy Scores...
print('accuracy score for bernouli:',accuracy_score(ytest,ypred_bb))
print('=========================================================')
print('accuracy score for multinomial:',accuracy_score(ytest,ypred_multi))
print('=========================================================')
print("accuracy score for gaussian:",accuracy_score(ytest,ypred_gaus))
print('=========================================================')
print('accuracy score for logistic regression:',accuracy_score(ytest,ypred_lr))
print('=========================================================')
print('accuracy score for random forest classifier:',accuracy_score(ytest,ypred_rfc))
