# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df=pd.read_csv(path)
print(df.head(5))
X=df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']]
print(X.head(4))
y=df.list_price
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)
# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
cols=X_train.columns
print(cols)
print(cols.shape[0])
print(X_train.shape[0])
fig,axes =plt.subplots(nrows = 3 , ncols = 3)
for i in range(0,3):
    for j in range(0,3):
        col =cols[i * 3 + j]
        axes[i,j]=plt.scatter(df[col],df.list_price)

# code ends here



# --------------
# Code starts here
corr=X_train.corr()
print(corr)
print(corr[corr>0.75].any())
print(corr[corr<-0.75].any())
X_train.drop(['play_star_rating','val_star_rating'],inplace=True,axis=1)
X_test.drop(['play_star_rating','val_star_rating'],inplace=True,axis=1)

# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
# Code starts here
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
mse=round(mean_squared_error(y_test,y_pred),3)
print(mse)
r2=r2_score(y_test, y_pred)
print(r2)
# Code ends here


# --------------
# Code starts here
residual =y_test-y_pred
print(residual)
plt.hist(residual)


# Code ends here


