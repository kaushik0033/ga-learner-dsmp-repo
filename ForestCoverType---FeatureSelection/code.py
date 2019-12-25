# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here
dataset =pd.read_csv(path)
dataset.iloc[:,0:5]
dataset.drop(['Id'],inplace=True,axis=1)
dataset.describe()

# read the dataset



# look at the first five columns


# Check if there's any column which is not useful and remove it like the column id


# check the statistical description



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 


#number of attributes (exclude target)
cols=dataset.columns.tolist()
size=len(cols[:-1])
y=dataset[cols[:-1]]
x =dataset.iloc[:,-1]
fig, ax = plt.subplots(figsize =(9, 7)) 
for i in range(5):
    sns.violinplot(ax = ax, x = y[cols[i]],  
                  y = x )

#x-axis has target attribute to distinguish between classes


#y-axis shows values of an attribute


#Plot violin for all attributes



# --------------
import numpy as np
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
subset_train=dataset.iloc[:,0:10]
data_corr=subset_train.corr()
sns.heatmap(data_corr)

correlation=data_corr.unstack().sort_values(kind='quicksort')
mask=((correlation>upper_threshold) & (correlation!=1.0)) | ((correlation<lower_threshold) & (correlation!=1.0))

corr_var_list= correlation[mask]
corr_var_list





# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)



# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,Y,test_size=0.2,random_state=0)
#X_train.drop(['Id'],axis=1,inplace=True)
#X_test.drop(['Id'],axis=1,inplace=True)
scaler=StandardScaler()
X_train_temp=scaler.fit_transform(X_train.iloc[:,0:10])
X_test_temp=scaler.transform(X_test.iloc[:,0:10])
X_train1=np.concatenate((X_train_temp,X_train.iloc[:,10:]),axis=1)
X_test1=np.concatenate((X_test_temp,X_test.iloc[:,10:]),axis=1)

scaled_features_train_df=pd.DataFrame(X_train1,columns=X_train.columns,index=X_train.index)
scaled_features_test_df=pd.DataFrame(X_test1,columns=X_test.columns,index=X_test.index)
#Standardized
#Apply transform only for continuous data
#scaled_features_train_df=pd.DataFrame(X_train1,columns=X_train.columns)
#scaled_features_test_df=pd.DataFrame(X_test1,columns=X_test.columns)
#Concatenate scaled continuous data and categorical



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(f_classif, percentile=90)
predictors=skb.fit_transform(X_train1, y_train)
scores=skb.scores_.tolist() 
Features=X_train.columns
dataframe=pd.DataFrame({'features':Features,'scores':scores})
dataframe.sort_values(by='scores',ascending=None,inplace=True)
top_k_predictors = list(dataframe['features'][:predictors.shape[1]])
top_k_predictors



# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
clf = LogisticRegression(random_state=0, multi_class='ovr')
clf1 = LogisticRegression(random_state=0, multi_class='ovr')
model_fit_all_features =clf1.fit(X_train,y_train)
predictions_all_features=model_fit_all_features.predict(X_test)
score_all_features=model_fit_all_features.score(X_test,y_test)
print(score_all_features)
model_fit_top_features=clf.fit(scaled_features_train_df[top_k_predictors],y_train)
predictions_top_features=model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
score_top_features=model_fit_top_features.score(scaled_features_test_df[top_k_predictors],y_test)
print(score_top_features)


