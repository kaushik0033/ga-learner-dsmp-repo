# --------------
import pandas as pd
from collections import Counter

# Load dataset
data=pd.read_csv(path)
print(data.isna().sum()/data.shape[0])
print(data.info())
print(data.describe())


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label=data.Activity
sns.countplot(label,data=data)
plt.xticks(rotation=90)
# plot the countplot



# --------------
# make the copy of dataset
data_copy = data.copy()

# Create an empty column 
data_copy['duration'] = ''

# Calculate the duration
duration_df = (data_copy.groupby([label[label.isin(['WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])], 'subject'])['duration'].count() * 1.28)
duration_df = pd.DataFrame(duration_df)

# Sort the values of duration
plot_data = duration_df.reset_index().sort_values('duration', ascending=False)
plot_data['Activity'] = plot_data['Activity'].map({'WALKING_UPSTAIRS':'Upstairs', 'WALKING_DOWNSTAIRS':'Downstairs'})


# Plot the durations for staircase use
plt.figure(figsize=(15,5))
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')
plt.title('Participants Compared By Their Staircase Walking Duration')
plt.xlabel('Participants')
plt.ylabel('Total Duration [s]')
plt.show()


# --------------
#exclude the Activity column and the subject column
feature_cols =data.select_dtypes(include='float64').columns
correlated_values=data[feature_cols].corr().stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2',0:'Correlation_score'})


#Calculate the correlation values
corr_var_list =pd.DataFrame(correlated_values)

#stack the data and convert to a dataframe
corr_var_list['abs_correlation']=corr_var_list.Correlation_score.abs()


#create an abs_correlation column
s_corr_list=corr_var_list.sort_values(by='abs_correlation')
top_corr_fields =s_corr_list[s_corr_list['abs_correlation']>0.8]

#Picking most correlated features without having self correlated pairs
top_corr_fields = top_corr_fields[top_corr_fields['Feature_1'] != top_corr_fields['Feature_2']].reset_index(drop=True)
print(top_corr_fields)



# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score



# Encoding the target variable
le = LabelEncoder()
data['Activity'] = le.fit_transform(data.Activity)
X = data.iloc[:,:-1]    
y = data.iloc[:,-1]

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


# Baseline model 
classifier = SVC()
clf = classifier.fit(X_train, y_train)
y_pred = clf.predict(X_test)
precision, recall, f_score, _ = error_metric(y_test, y_pred, average = 'weighted')
model1_score = accuracy_score(y_test, y_pred)
print(model1_score)
print(precision, recall, f_score)



# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc=LinearSVC(C=0.01,penalty='l1',dual=False,random_state=42)
lsvc.fit(X_train,y_train)
model_2=SelectFromModel(lsvc,prefit=True)
new_train_features=model_2.transform(X_train)
new_test_features =model_2.transform(X_test)
# Feature selection using Linear SVC
classifier_2=SVC()
clf_2=classifier_2.fit(new_train_features,y_train)
y_pred_new=clf_2.predict(new_test_features)
precision, recall, f_score, _ = error_metric(y_test, y_pred_new, average = 'weighted')
model2_score = accuracy_score(y_test, y_pred_new)
print(model2_score)
print(precision, recall, f_score)
# model building on reduced set of features




# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV
parameters ={'C': [100,20,1,0.1], 'kernel': ('linear', 'rbf')}
selector=GridSearchCV(SVC(),parameters,scoring='accuracy')
selector.fit(new_train_features,y_train)
print(selector.best_params_)
means=selector.cv_results_['mean_test_score']
stds=selector.cv_results_['std_test_score']
print(means)
print(stds)
# Set the hyperparmeters
classifier_3=SVC(C=100)
clf_3=classifier_3.fit(new_train_features,y_train)
y_pred_final=clf_3.predict(new_test_features)
precision, recall, f_score, _ = error_metric(y_test, y_pred_final, average = 'weighted')
model3_score = accuracy_score(y_test, y_pred_final)
print(model3_score)
print(precision, recall, f_score)

# Usage of grid search to select the best hyperparmeters



    

# Model building after Hyperparameter tuning





