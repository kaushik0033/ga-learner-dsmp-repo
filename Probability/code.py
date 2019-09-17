# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
p_a=df[df['fico'].astype(float) >700].shape[0]/df.shape[0]
p_b=df[df['purpose']=='debt_consolidation'].shape[0]/df.shape[0]
df1=df[df['purpose']=='debt_consolidation']
p_b_a=df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]
#print(p_a.head(5))
#print(p_b.head(5))
p_a_b=p_b_a*p_a/p_b

result=p_a_b==p_a
print(result)
# code ends here


# --------------
# code starts here
prob_lp=df[df['paid.back.loan']=='Yes'].shape[0]/df.shape[0]
prob_cs=df[df['credit.policy']=='Yes'].shape[0]/df.shape[0]
new_df=df[df['paid.back.loan']=='Yes']
prob_pd_cs=new_df[new_df['credit.policy']=='Yes'].shape[0]/new_df.shape[0]
bayes=prob_pd_cs*prob_lp/prob_cs
print(round(bayes,4))
# code ends here


# --------------
fig, ax = plt.subplots() 
# count the occurrence of each class 
data = df['purpose'].value_counts() 
# get x and y data 
points = data.index 
frequency = data.values 
# create bar chart 
ax.bar(points, frequency) 

df1=df[df['paid.back.loan']=='No']
data = df1['purpose'].value_counts() 
# get x and y data 
points = data.index 
frequency = data.values 
# create bar chart 
ax.bar(points, frequency) 



# --------------
# code starts here
inst_median=df['installment'].median()
inst_mean=df['installment'].mean()

fig,ax=plt.subplots()
ax.hist(inst_median)
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')
ax.hist(inst_mean)
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')
# code ends here


