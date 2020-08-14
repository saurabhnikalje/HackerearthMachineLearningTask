from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
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
train, test, train_labels, test_labels = train_test_split(xtrain,target,test_size = 0.3, random_state = None,shuffle=True)


from sklearn.svm import SVC
#Creating and Initializing the SVM Classifier
clf = KNeighborsClassifier()
clf.fit(train,train_labels)
prediction = clf.predict(test)


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


