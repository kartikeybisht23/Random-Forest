import pandas as pd
import numpy as np
features=pd.read_csv("temperaure.csv")
features.head()
features.describe()
#one hot encoding
features=pd.get_dummies(features)
features.head()

#converting it in features
import numpy as np
labels=np.array(features["actual"])
features=features.drop("actual",axis=1)
features_list=list(features.columns)
features_list=np.array(feature_list)

#scikit learn to split
from sklearn.model_selection import train_test_split
train_features,test_features,train_labels,test_labels=train_test_split(features,labels,test_size=0.25,random_state=42)


# import random model and fitting it
from sklearn.ensemble import RandomForestRegressor
rf_m=RandomForestRegressor(n_estimators=1000,random_state=42)
rf_m.fit(train_features,train_labels)

predictions=rf_m.predict(test_features)
errors=abs(predictions-test_labels)
print(f"mean absolute error={round(np.mean(errors),2)}degrees")

# now let's determine the performance matrix
mpa=100*(errors/test_labels)
accuracy=100-np.mean(mpa)
print(f"accuracy:{accuracy}")

import matplotlib.pyplot as plt
from sklearn import tree
y=rf_m.estimators_[1]
plt.figure(figsize=(12,10))
tree.plot_tree(y)

#use date time for creating date objects for samples
#using date time for plotting objects
import datetime
#dates of training values
months=list(features.loc[: , "month"])
days=list(features.loc[:,"day"])
years=list(features.loc[:,"year"])
# converting above into date time object
dates=[str(year)+"-"+str(month)+"-"+str(day) for year,month,day in zip(years,months,days)]
dates=[datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]
true_data=pd.DataFrame(data={"date":dates,"actual":labels})
#dates of prediction
months=list(test_features.loc[:,"month"])
days=list(test_features.loc[:,"day"])
years=list(test_features.loc[:,"year"])
test_dates=[str(year)+"-"+str(month)+"-"+str(day) for year,month,day in zip(years,months,days)]
test_dates=[datetime.datetime.strptime(date,"%Y-%m-%d") for date in test_dates]
#dataframe of predcitions with date
prediciton_data=pd.DataFrame(data={"test_dates":test_dates,"predictions":predictions})
#plotting the acutal values:
plt.plot(true_data["date"],true_data["actual"],"b-",label="actual")
#plotting the predicted values:
plt.plot(prediciton_data["test_dates"],prediciton_data["predictions"],'ro',label="predcitions")
plt.xticks(rotation="60")
plt.legend()
#graph label
plt.xlabel("dates");plt.ylabel("Maximum temperatures");plt.title("actual and precited values")
