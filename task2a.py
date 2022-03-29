import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier



def compute_var(world, features, mean_list):
    var_list = []
    for feature in features:
        datalist = world[feature].astype(float)
        var_list.append(np.var(datalist))
    return var_list

def compute_mean(world, features):
    mean_list = []
    for feature in features:
        datalist = world[feature].astype(float)
        mean_list.append(np.mean(datalist))
    return mean_list
            

def compute_median(world, feature):
    data_num = []
    for row in range(0, 264):
        if world[feature][row] != '..':
            data_num.append(float(world[feature][row]))
        else:
            continue
    return np.median(data_num)


##load in the data
world=pd.read_csv('world.csv')
life=pd.read_csv('life.csv')

life_list = []
for row in range(len(life)):
    life_list.append([life['Country Code'][row], life['Life expectancy at birth (years)'][row]])

world["Life expectancy at birth (years)"] = ""


#include the life data to the world data
for row in range(len(life)):
    for life_data in life_list:
        if world['Country Code'][row] == life_data[0]:
            world['Life expectancy at birth (years)'][row] = life_data[1]
            continue

##get just the features
features = []
for feature in world:
    if feature != 'Life expectancy at birth (years)' and feature != 'Country Name' and feature != 'Time' and feature != 'Country Code':
        features.append(feature)

        
#dic to make task2a.csv file        
task2a_fields = dict()

median_list = []
feature_list = []

#replace '..' with median imputation
for feature in features:
    for row in range(0, 264):
        if world[feature][row] == '..':
            world[feature][row] = compute_median(world,feature)
        else:
            world[feature][row] = world[feature][row]
            
    #attain the list for the final task2a.csv
    median_list.append(compute_median(world,feature))
    feature_list.append(feature)
    
            
world.drop(world.tail(5).index,inplace=True)
world.to_csv('finally.csv')


data=world[features].astype(float)


##get just the class labels
classlabel=world['Life expectancy at birth (years)']

##randomly select 66% of the instances to be training and the rest to be testing
X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=0.66, test_size=0.34, random_state=100)

#normalise the data to have 0 mean and unit variance using the library functions.  This will help for later
#computation of distances between instances
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#decisiontree accuracy
dt = DecisionTreeClassifier(criterion="entropy",random_state=100, max_depth=4)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
print('Accuracy of decision tree:{}'.format(round(accuracy_score(y_test, y_pred),3)))

#k-nn at k = 5
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print('Accuracy of k-nn (k=5):{}'.format(round(accuracy_score(y_test, y_pred),3)))


#k-nn at k = 10
knn10 = neighbors.KNeighborsClassifier(n_neighbors=10)
knn10.fit(X_train, y_train)
y_pred=knn10.predict(X_test)
print('Accuracy of k-nn (k=10):{}'.format(round(accuracy_score(y_test, y_pred),3)))


mean_list = compute_mean(world, features)
var_list = compute_var(world, features, mean_list)


task2a = pd.DataFrame({'feature':feature_list, 'median': median_list, 'mean':mean_list, 'variance':var_list})
task2a.to_csv('task2a.csv',index=False)
