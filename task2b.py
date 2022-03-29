import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans

import scipy.stats as stats
from scipy.stats import chi2_contingency


def compute_chi_value(data, column):

    cont_table = pd.crosstab(column,data['Cluster'])
    chi2_val, p, dof, expected = stats.chi2_contingency(cont_table.values, correction=False)
    #print('Chi2 value: ',chi2_val)
    if(p<0.05) : 
        #print('Null hypothesis rejected, p value: ', p)
        return chi2_val
    else :
        #print('Null hypothesis accepted, p value: ', p)
        return 0
def compute_median(world, feature):
    data_num = []
    for row in range(len(world)):
        if world[feature][row] != '..':
            data_num.append(float(world[feature][row]))
        else:
            continue
    return np.median(data_num)

def find_the_top_four(data,inter_data,classlabel):
    list_chi = []
    best4 = []
    for feature in inter_data:
        if feature != 'Cluster':
            column_data = inter_data[feature]
            #column_data = np.reshape(column_data, (len(column_data),-1))
            list_chi.append([compute_chi_value(data, column_data), feature])
    for chi in list_chi[-4:]:
        best4.append(chi[1])
    best4_df = pd.DataFrame(columns=best4)
    for feature in best4:
        best4_df[feature] = inter_data[feature]
    return best4_df
def compute_accuracy(data, classlabel):
    X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=0.66, test_size=0.34, random_state=100)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    return round(accuracy_score(y_test, y_pred),3)

def compute_pca_accuracy(data, classlabel):
    ##randomly select 66% of the instances to be training and the rest to be testing
    X_train, X_test, y_train, y_test = train_test_split(data,classlabel, train_size=0.66, test_size=0.34, random_state=100)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)


    # to compute the 4 principal values (accuracy using PCA)
    pca = decomposition.PCA(n_components = 4)
    pca.fit(X_train)
    pca.fit(X_test)
    x_pca = pca.transform(X_train)
    x_test_pca = pca.transform(X_test)


    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_pca, y_train)
    y_pred=knn.predict(x_test_pca)
    print('Accuracy of PCA:{}'.format(round(accuracy_score(y_test, y_pred),3)))


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


world.drop(world.tail(5).index,inplace=True)
#replace '..' with median imputation
for feature in features:
    for row in range(len(world)):
        if world[feature][row] == '..':
            world[feature][row] = compute_median(world,feature)
        else:
            world[feature][row] = world[feature][row]
            
    #attain the list for the final task2a.csv
    median_list.append(compute_median(world,feature))
    feature_list.append(feature)
    
data=world[features].astype(float)


kmeans = KMeans(n_clusters=3).fit(data)
kmeans.fit(data)
clusters = kmeans.cluster_centers_
y_km = kmeans.fit_predict(data)
inter_data = data

#make the interaction 190 features + cluster feature + original 20 features
done_features = []
for feature in features:
    for feature1 in features:
        if feature1 not in done_features:
            if feature != feature1:
                inter_data[feature+"*"+feature1] = data[feature]*data[feature1]
    done_features.append(feature)
inter_data['Cluster'] = y_km

data.to_csv('finally.csv')


##get just the class labels
classlabel=world['Life expectancy at birth (years)']

#get best 4 from the chi squared values and compute the accuracy
best4 = find_the_top_four(data,inter_data,classlabel)
print('Accuracy of feature engineering:{}'.format(compute_accuracy(best4, classlabel)))



#get first from from data and compute the accuracy
first4_df = pd.DataFrame(columns=features[0:4])
for feature in features[0:4]:
    first4_df[feature] = data[feature]
print('Accuracy of first four features:{}'.format(compute_accuracy(first4_df, classlabel)))

compute_pca_accuracy(data, classlabel)