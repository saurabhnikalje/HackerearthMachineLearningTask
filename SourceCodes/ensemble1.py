import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

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
#To deal with Not a number values.
x = np.nan_to_num(xtrain)

#Create Sub models
estimator = []
model1 = RandomForestClassifier()
estimator.append(('RandomForest',model1))
model2 = KNeighborsClassifier()
estimator.append(('KNN',model2))
ensemble = VotingClassifier(estimator,voting='hard')
clf = ensemble.fit(x,target)
print(clf)

#Save the trained model in local directory.
import pickle
filename = 'finalmodel.sav'
pickle.dump(clf,open(filename,'wb'))


#Test dataset
url2 = r'C:\Users\lenovo\Desktop\Hack\Dataset\Test.csv';
testdata = pd.read_csv(url2);
test = testdata.loc[:,['X_3','X_5','X_11','X_13','X_14','X_2','X_4','X_6','X_7','X_9']]
test = np.nan_to_num(test)

#Load the model
loaded_model = pickle.load(open(filename,'rb'))
prediction = loaded_model.predict(test)
print(prediction)

#np.savetxt(r'C:\Users\lenovo\Desktop\predictions.csv',prediction,delimiter=',')

p = pd.DataFrame()
p.to_csv(prediction,)