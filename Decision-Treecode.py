# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
data=pd.read_csv(path)
X=data.drop(columns=['customer.id','paid.back.loan'])
y=data['paid.back.loan']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid=y_train.value_counts()
plt.plot(kind='bar',x=fully_paid)
plt.show()
# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
X_train['int.rate']=X_train['int.rate'].str.replace('%','').astype('float')
X_train['int.rate']=X_train['int.rate']/100
X_test['int.rate']=X_test['int.rate'].str.replace('%','').astype('float')
X_test['int.rate']=X_test['int.rate']/100
num_df=X_train.select_dtypes(exclude='object')
cat_df=X_train.select_dtypes(include='object')
# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
cols=num_df.columns
fig,axes=plt.subplots(nrows=9,ncols=1)
for i in range(0,9):
    sns.boxplot(x=y_train,y=num_df[cols[i]])
#X_train.shape
#X_train.iloc[0:9]
# Code ends here


# --------------
# Code starts here
cols=cat_df.columns
fig,axes=plt.subplots(nrows=2,ncols=2)
for i in range(0,2):
    for j in range(0,2):
        sns.countplot(x=X_train[cols[i*2+j]],hue=y_train,ax=axes[i,j])
# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier
catcols=cat_df.columns
# Code starts here
for i in range(len(cat_df.columns)):
    X_train[catcols[i]].fillna('NA',inplace=True)
    X_test[catcols[i]].fillna('NA',inplace=True)

le=LabelEncoder()
X_train[catcols] = X_train[catcols].apply(le.fit_transform)
X_test[catcols]=X_test[catcols].apply(le.fit_transform)
y_train=y_train.replace('No',0).replace('Yes',1)
y_test=y_test.replace('No',0).replace('Yes',1)
model=DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)
train_score=model.score(X_train,y_train)
acc=model.score(X_test,y_test)
print(train_score)
print(acc)
# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model_2=DecisionTreeClassifier(random_state=0)
p_tree=GridSearchCV(estimator=model_2,param_grid=parameter_grid,cv=5)
p_tree.fit(X_train,y_train)
acc_2=p_tree.score(X_test,y_test)
print(acc_2)
# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data=tree.export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True , class_names=['loan_paid_back_yes','loan_paid_back_no'] )
graph_big=pydotplus.graph_from_dot_data(dot_data)
Image(graph_big)
# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


