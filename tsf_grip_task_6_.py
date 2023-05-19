
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#importing data
df=pd.read_csv("/content/drive/MyDrive/TSP-GRIP/Iris.csv")

"""Exploratory data analysis"""

df.shape

df.head(10)

df.info()

df.describe()

df.isna().sum()

df=df.dropna()

df=df.drop_duplicates()

#removing unnecessary features
df=df.drop(['Id'],axis=1)

print(df['Species'].unique())

"""Identifying the independent and dependent variables"""

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)

"""Fitting decision tree model"""

dt=DecisionTreeClassifier(criterion="entropy",random_state=0)
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

"""Evaluating the model"""

from sklearn import metrics 
cm_dt=metrics.confusion_matrix(y_test,y_pred)
plt.figure(figsize=(1.5,1.5))
sns.heatmap(cm_dt, linewidths=2, square = True, cmap = 'Blues_r')
plt.title("Accuracy matrix")
plt.show()
print("\nAccuracy score : ",metrics.accuracy_score(y_test,y_pred))

"""Visualizing the tree"""

from sklearn import tree
fig,ax=plt.subplots(figsize=(10,10))
tree.plot_tree(dt,ax=ax,fontsize=6,feature_names=df.columns,class_names=['setosa', 'versicolor', 'virginica'],filled=True)
plt.show()

"""Exploring model with new data"""

#input new data
new_data=[]
for i in range(4):
  #a="sepal length"
  cols=["sepal length","sepal width","petal length","petal width"]
  value=float(input(f"Enter the {str(cols[i])} :"))
  new_data.append(value)

prediction=dt.predict(np.reshape(new_data,(1, -1)))
print(prediction)
