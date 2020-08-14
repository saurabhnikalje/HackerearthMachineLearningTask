from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Import Training Dataset
url1 =r'C:\Users\lenovo\Desktop\Hack\Dataset\Train.csv';
dataset = pd.read_csv(url1);
print(dataset.shape)

data = pd.DataFrame(dataset)
print(data.dtypes)
count = data.groupby('MULTIPLE_OFFENSE').size()
print(count)

xtrain = data.loc[:,['X_3','X_5','X_11','X_13','X_14','X_2','X_4','X_6','X_7','X_9']]
target = data.MULTIPLE_OFFENSE
x = np.nan_to_num(xtrain)

from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(xtrain,target,test_size = 0.30, random_state = None,shuffle=True)


#Standardized the data
scaler= StandardScaler()
scaled_data = scaler.fit_transform(train)
print("***********The Data after Standardization using StandardScalar********** ")
#Standardised data had Nan(Not a number) values and infinite values which cannot be used for PCA.
#So nan_to_num is used in which Nan is replaced by zero and infinte values by largest finite value.
scaled_data = np.nan_to_num(scaled_data)
#Check  the results after this function.
print(np.any(np.isnan(scaled_data)))
print(np.all(np.isfinite(scaled_data)))
print(scaled_data)
print(np.shape(scaled_data))


#MODEL CREATION
from sklearn.naive_bayes import GaussianNB
gnb = LinearRegression()
model = gnb.fit(scaled_data,train_labels)
prediction = model.predict(test)
print(prediction)


from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,prediction))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(test_labels, prediction)
print("Confusion Matrix:")
print(result)
result1 = classification_report(test_labels, prediction)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(test_labels,prediction)
print("Accuracy:",result2)




















'''

'''

'''
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
# define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=9)
rfe.fit(x, y)
# summarize all features
for i in range(x.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))



data.hist()
plt.show()

print(np.any(np.isnan(xtest)))
print(np.all(np.isfinite(xtest)))
'''


'''#Data Pre Processing
from sklearn.preprocessing import  MinMaxScaler

datascaler = MinMaxScaler(feature_range=(0,1))
data_rescaled = datascaler.fit_transform(xtrain)
#print(data_rescaled[0:10])'''


#print(count(test['MULTIPLE_OFFENSE']))