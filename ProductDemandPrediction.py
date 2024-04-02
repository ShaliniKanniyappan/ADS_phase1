import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
data = pd.read_csv("C:/Users/ss/Documents/ibm project/PoductDemand.csv")
data.head()
data.isnull().sum()
data = data.dropna()
data.isnull().sum()
one_hot_encoded_data = pd.get_dummies(data, columns = ['Store ID', 'Units Sold']) 
print(one_hot_encoded_data)
fig = px.scatter(data, x="Units Sold", y="Total Price",size='Units Sold')
fig.show()
print(data.corr())
correlations = data.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True,fmt=".2f", linewidth=.5)
plt.show()
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm 
regr=LinearRegression() 
regr.fit(xtrain, ytrain)
print("r2 score=",sm.r2_score(ytest,y_pred))
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
y_pred = model.predict([[190.2375,234.4125]]) 
print("units sold(predicted): % d\n"% y_pred)  
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf=DecisionTreeClassifier(max_depth=5)
clf1=clf.fit(xtrain,ytrain)
y_predict=clf1.predict(xtest)
fig=plt.figure()
tree.plot_tree(clf1)
plt.show()
fig.savefig("decision_tree.png")
from sklearn.metrics import accuracy_score
print("Accuracy:")
accuracy=accuracy_score(ytest,y_predict)
print(accuracy*100,"%")
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
conf_matrix = metrics.confusion_matrix(ytest,y_predict) 
print("Confusion Matrix – Decision Tree") 
print(conf_matrix) 