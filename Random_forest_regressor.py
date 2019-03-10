# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:05:33 2018

@author: hp
"""

#Random forest regression

#IMPORTING LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#IMPORTING DATASET
dataset=pd.read_csv("Position_Salaries.csv")
features=dataset.iloc[:,1:2].values
labels=dataset.iloc[:,2].values

#CREATING DECISION TREE MODEL
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(features,labels)

#visualising
plt.scatter(features,labels,color='red')
plt.plot(features,regressor.predict(features),color='blue')
plt.title('position vs salary')
plt.xlabel('level')
plt.ylabel('sal')
plt.show()

#visualising with smoother curve
features_grid=np.arange(min(features),max(features),0.01)
features_grid=features_grid.reshape((len(features_grid),1))
plt.scatter(features,labels,color='red')
plt.plot(features_grid,regressor.predict(features_grid),color='blue')
plt.title('position vs salary')
plt.xlabel('level')
plt.ylabel('sal')
plt.show()

#predicting a value
regressor.predict(6.5)