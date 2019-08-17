# --------------
import numpy as np
import pandas as pd
from scipy.stats import mode 
data=pd.read_csv(path)
#data.shape
#data.dtypes
bank=pd.DataFrame(data)
#bank.head(5)
categorical_var=bank.select_dtypes(include='object')
print(categorical_var)
numerical_var=bank.select_dtypes(include='number')
print(numerical_var)


# --------------
# code starts here
banks=pd.DataFrame(data=bank)
banks.drop('Loan_ID',axis=1,inplace=True)
banks.head(2)
print(banks.isnull().sum())
bank_mode=banks.mode()
banks.fillna(str(bank_mode),inplace=True,axis=1)
print(banks.isnull().sum())
#code ends here


# --------------
# Code starts here
banks.head(2)
#banks['LoanAmount']= banks['LoanAmount'].astype(int)
avg_loan_amount=pd.pivot_table(banks,index=['Gender','Married','Self_Employed'],values='LoanAmount')
print(avg_loan_amount)
# code ends here


# --------------
# code starts here
banks.head(2)
loan_approved_se=banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')]
print(loan_approved_se.head(2))
loan_approved_nse=banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')]
print(loan_approved_nse)
percentage_se=(len(loan_approved_se)/614)*100
percentage_nse=(len(loan_approved_nse)/614)*100
print(percentage_se)
print(percentage_nse)
# code ends here


# --------------
# code starts here
#banks.head(2)
loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12 )


big_loan_term=len(loan_term[loan_term>=25])

print(big_loan_term)



# code ends here


# --------------
# code starts here
loan_groupby=banks.groupby(['Loan_Status'])

loan_groupby=loan_groupby[['ApplicantIncome', 'Credit_History']]
loan_groupby.head(2)
mean_values=loan_groupby.mean()
mean_values

# code ends here


