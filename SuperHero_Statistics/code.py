# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data=pd.read_csv(path)
#Code starts here 
data['Gender'].replace('-','Agender',inplace=True)
gender_count=data.Gender.value_counts()
gender_count.plot(kind='Bar')




# --------------
#Code starts here
alignment=data.Alignment.value_counts()
alignment.plot(kind='Bar')
plt.xlabel('Character Alignment')


# --------------
#Code starts here
sc_df=data[['Strength','Combat']]
sc_covariance=sc_df.cov().iloc[1,0].round(2)
sc_strength=sc_df.Strength.std().round(2)
sc_combat=sc_df.Combat.std().round(2)
sc_pearson=(sc_covariance/(sc_strength*sc_combat)).round(2)
ic_df=data[['Intelligence','Combat']]
ic_covariance=ic_df.cov().iloc[1,0].round(2)
ic_intelligence=ic_df.Intelligence.std().round(2)
ic_combat=ic_df.Combat.std().round(2)
ic_pearson=(ic_covariance/(ic_intelligence*ic_combat)).round(2)


# --------------
#Code starts here
total_high=data.Total.quantile(q=0.99)
print(total_high)
super_best=data[data['Total']>total_high]
super_best_names=list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
fig,([ax_1,ax_2,ax_3])=plt.subplots(1,3,figsize=(10,15))
data['Intelligence'].plot(kind='box', ax=ax_1, legend=True)
data['Speed'].plot(kind='box', ax=ax_2, legend=True)
data['Power'].plot(kind='box', ax=ax_3, legend=True)

#ax_1.set_xlabels('Intelligence')
#ax_2=plt.boxplot(data.Speed)
#ax_2.set_xlabels('Speed')
#ax_3=plt.boxplot(data.Power)
#ax_3.set_xlabels('Power')



