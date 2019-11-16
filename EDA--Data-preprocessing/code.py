# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data=pd.read_csv(path)
data.Rating.hist()
data=data[data['Rating']<=5]
data.Rating.hist()
#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
print(data.isnull().sum())
print(data.isnull().count())
percent_null=total_null/ data.isnull().count()
missing_data=pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'] )
print(missing_data)

data.dropna(inplace=True)
total_null_1=data.isnull().sum()
print(total_null_1)
percent_null_1=total_null_1/ data.isnull().count()
missing_data_1=pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'] )

print(missing_data_1)


# --------------

#Code starts here
sns.catplot(x="Category",y="Rating",data=data, kind="box",height=10,orient=90)
#sns.set_title('Rating vs Category [BoxPlot]')

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data.Installs.value_counts())
data['Installs']=data.Installs.str.replace(',','')
data['Installs']=data.Installs.str.replace('+','')
data['Installs']=data['Installs'].astype(int)

le=LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
sns.regplot(x="Installs", y="Rating",data=data)
#Code ends here



# --------------
#Code starts here
print(data.Price.value_counts())
data['Price']=data.Price.str.replace('$','')
data['Price']=data['Price'].astype(float)

le=LabelEncoder()
data['Price'] = le.fit_transform(data['Price'])
sns.regplot(x="Price", y="Rating",data=data)



#Code ends here


# --------------

#Code starts here
#print(data.Genres.unique())
#print(data[data.Genres.str.contains(';')].head(10))
data['Genres']=data.Genres.str.split(';',expand=True)
#print(data.head(5))
gr_mean=data.groupby(['Genres'],as_index=False)['Rating'].mean()
print(gr_mean.describe())
print(gr_mean.min())
print(gr_mean.max())
gr_mean=gr_mean.sort_values('Rating')
#print(gr_mean.head(50))
#gr_mean=gr_mean.sort_values(1) 
print(gr_mean.head(),gr_mean.tail())
#print(gr_mean.iloc[[0, -1]])
#print(pd.concat([gr_mean.head(1), gr_mean.tail(1)])
#Code ends here


# --------------
#Code starts here
print(data.head(2))
data['Last Updated']= pd.to_datetime(data['Last Updated'], infer_datetime_format=True)
max_date=max(data['Last Updated'])
print(data.head(2))
print(max_date)
data['Last Updated Days']=(max_date-data['Last Updated']).dt.days
print(data.head(2))
sns.regplot(x=data['Last Updated Days'],y='Rating',data=data)
#sns.set_title('Rating vs Last Updated [RegPlot]')
#Code ends here


