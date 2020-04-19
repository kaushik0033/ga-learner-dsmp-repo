#!/usr/bin/env python
# coding: utf-8

# ## Capstone Project - Mahindra First Choice Services -
# ### Loading libraries...

# In[1]:


#import libraries
from datetime import datetime, timedelta,date
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report,confusion_matrix
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
pd.plotting.register_matplotlib_converters()
sns.mpl.rc('figure',figsize=(10, 5))
plt.style.use('ggplot')
sns.set_style('darkgrid')
# Standard plotly imports
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go_offline
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# In[2]:


# customize print function to format the text
def printColorText(text,value=''):
    print("\33[01m {}\033[00m" .format(text),value,'\n')

# check statistics on given dataset
def displayDataSetInfo(data_set,name=''):
    printColorText('Shape of {} data -'.format(name),data_set.shape)
    printColorText('Display top 10 rows of {} data - \n'.format(name),data_set.head(10))
    printColorText('\n Display summary of statistics for {} data - \n\n'.format(name),data_set.describe())
    printColorText('\n Quick overview of {} data - '.format(name))
    print(data_set.info())
    printColorText('\n Detect % of missing values in the given {} dataset - \n\n'.format(name),round(data_set.isnull().sum()/data_set.shape[0],2))

    
# cross validate pipeline with multiclassfier 
def cross_validate_with_multiclassifier(clfs,pipe,X_train, y_train):
    for clf in clfs:
        pipe.set_params(classifier = clf)
        scores = cross_validate(pipe, X_train, y_train)
        print('===================================')
        classifier=clf
        print(str(classifier))
        print('===================================')
        for key, values in scores.items():
            print(key,' mean ', values.mean())
            print(key,' std ', values.std())

            #function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# #### Load dataset provided by Mahindra...

# In[3]:


df_plantmaster_data=pd.read_excel('D:/Grayatom/Mahindra/file/Plant Master.xlsx')
df_jobtime_data=pd.read_csv('D:/Grayatom/Mahindra/file/JTD.csv')
df_customer_data=pd.read_excel('D:/Grayatom/Mahindra/file/Customer_Data.xlsx')
df_transaction_data=pd.read_csv('D:/Grayatom/Mahindra/file/Final_invoice.csv')
# Renaming columns
df_plantmaster_data.columns=df_plantmaster_data.columns.str.lower().str.replace(' ','_').str.replace('(','').str.replace(')','').str.replace('.','')
df_jobtime_data.columns=df_jobtime_data.columns.str.lower().str.replace(' ','_').str.replace('(','').str.replace(')','').str.replace('.','')
df_customer_data.columns=df_customer_data.columns.str.lower().str.replace(' ','_').str.replace('(','').str.replace(')','').str.replace('.','')
df_transaction_data.columns=df_transaction_data.columns.str.lower().str.replace(' ','_').str.replace('(','').str.replace(')','').str.replace('.','')

# Change Dates field into DateTime 
df_transaction_data[["invoice_date", "jobcard_date"]] = df_transaction_data[["invoice_date", "jobcard_date"]].apply(pd.to_datetime)


# #### Display Plant Dataset

# In[4]:


displayDataSetInfo(df_plantmaster_data,'Plant Master')


# In[7]:


df_plantmaster_data.drop(columns=['valuation_area','vendor_number_plant','name_2','po_box','house_number_and_street','factory_calendar'],axis=1,inplace=True)


# ### First observation from plant master dataset
# #### ACTION :DROP name_2,Valuation_area, Vendor_number_plant and PO box  - these are redundant informative fields, 
# #### ACTION :DROP Not Significant/userful Fields - house_number_and_street,factory_calendar 
# 
# ### Useful fields - 
#     1.Plant  - NO MISSING
#     2.name_1 - NO MISSING
#     3.city   - NO MISSING
#     4.postalcode- NO MISSING
#     5.state-      NO MISSING
#     6.sales orgnization - 
#     #### Sales Organization -0.01 , Categorical field 

# In[6]:


print(set(df_plantmaster_data.sales_organization))
print(df_plantmaster_data.sales_organization.isnull().value_counts())
df_plantmaster_data[df_plantmaster_data.sales_organization.isnull()]


# In[7]:


# Unique plant values - 438
df_plantmaster_data.plant.nunique()


# In[8]:


# filling 5 missing value by calculating most frequent
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer = imputer.fit(df_plantmaster_data)
df_plantmaster_data_imputed = imputer.transform(df_plantmaster_data)


# In[9]:


df_plantmaster_data=pd.DataFrame(df_plantmaster_data_imputed,columns=['plant_code','plant_name','customer_no_plant',                                                                      'postal_code','city','sales_organization','state'])

df_plantmaster_data.head(5)


# In[15]:


df_plantmaster_data.to_csv("D:/Grayatom/Mahindra/file/CleanData/plant_master_clean.csv")


# In[69]:


# Plant Data visualization
df_organization_group=df_plantmaster_data.groupby(['sales_organization','state'])['plant_code'].count()
df_organization_group = pd.DataFrame(df_organization_group)
df_organization_group_count=df_organization_group.reset_index().sort_values('plant_code', ascending=False)

plot_data = [
    go.Histogram(
        x=df_organization_group_count['state'],
        y=df_organization_group_count['plant_code'],
    )
]
plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Frequency of Sales Org by State'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)

print("#### Observation")

print("#### First Rank:  MFCD plants are setup widely through out India. ")
print("#### FIVE State : Rajasthan ,UP, Maharastra,Tamil Nadu,MP ")
print("#### North, Center and West zones:  Multiple Sales Org. are active - favourtie region ")
print("#### East zone:  There are two sales org. active MFCD and MFCC")


# In[86]:


df_organization_city_group=df_plantmaster_data.groupby(['sales_organization','city'])['plant_code'].count()
df_organization_city_group = pd.DataFrame(df_organization_city_group)

df_organization_city_group_count=df_organization_city_group.reset_index().sort_values('plant_code', ascending=False)
plot_data = [
    go.Scatter(
        x=df_organization_city_group_count[:10]['city'],
        y=df_organization_city_group_count[:10]['plant_code'],
    )
]
plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Frequency of Sales Org by City'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)
print("#### Observation")
print("#### 5 plants in City - Pune,Kolkata and Nashik ")
print("#### 4 plants in City - Indore")
print("#### 3 plants in City - Meerut,Aurangabad,Agra,Bhopal,Ranchi,Patna,Coimmbatore")


# ### Display Jobcard Dataset

# In[82]:


displayDataSetInfo(df_jobtime_data,'Job Time')


# In[89]:


df_jobtime_data.drop(columns=['unnamed:_0'],axis=1,inplace=True)


# In[88]:


df_jobtime_data['description'].isnull().sum()


# In[10]:


print(df_jobtime_data.query('material!=material').nunique())
print(df_jobtime_data.query('target_quantity_uom !=target_quantity_uom').nunique())


# In[90]:


df_jobtime_data.dropna(subset=['material','description','target_quantity_uom'],inplace=True)


# In[12]:


df_jobtime_data_subset=df_jobtime_data[df_jobtime_data.labor_value_number.notnull()]


# In[138]:


#df_jobtime_data_subset.description.unique().shape --905
#df_jobtime_data_subset.labor_value_number.unique().shape --41176
df_jobtime_data_labor_value=df_jobtime_data.query('labor_value_number==labor_value_number')


# In[93]:


# Drop labor value number as description field covers task name
df_jobtime_data.drop(['labor_value_number'],inplace=True,axis=1)


# In[92]:


df_mahindra_parts=df_jobtime_data[df_jobtime_data.description.str.contains("Mahindra")]


# In[22]:


df_mahindra_parts.to_csv("D:/Grayatom/Mahindra/file/CleanData/mahindra_parts.csv")


# In[91]:


job_value_lessthanzero=df_jobtime_data.query('net_value<0.0')
job_value_greaterthanzero=df_jobtime_data.query('net_value>0.0')
job_value_zero=df_jobtime_data.query('net_value==0.0')


# In[32]:


job_value_lessthanzero.to_csv("D:/Grayatom/Mahindra/file/CleanData/jobtime_lessthanzero.csv")


# In[33]:


job_value_zero.to_csv("D:/Grayatom/Mahindra/file/CleanData/jobtime_zero.csv")


# In[34]:


job_value_greaterthanzero.head(5)


# In[135]:


df_transaction_Parts_others=df_transaction_data.query('cust_type!="Retail"').merge(df_jobtime_data,how='inner',left_on='job_card_no',right_on='dbm_order')
df_transaction_Parts_others.drop(columns=['dbm_order','order_item','print_status','plant_name1'],inplace=True)
df_transaction_Parts_others.head(5)
#df_transaction_Parts_others.to_csv("D:/Grayatom/Mahindra/file/CleanData/transaction_othercusttype.csv")


# In[136]:


df_transaction_Parts_retail=df_transaction_data.query('cust_type=="Retail"').merge(df_jobtime_data,how='inner',left_on='job_card_no',right_on='dbm_order')
df_transaction_Parts_retail.drop(columns=['dbm_order','order_item','print_status','plant_name1'],inplace=True)
df_transaction_Parts_retail.head(5)


# In[140]:


top20_mostUsage_task_oth=df_transaction_Parts_others.groupby('description')['job_card_no'].count().reset_index().sort_values(by='job_card_no',ascending=False).head(20)

top20_mostUsage_task=df_transaction_Parts_retail.groupby('description')['job_card_no'].count().reset_index().sort_values(by='job_card_no',ascending=False).head(20)

plot_data = [
    go.Scatter(
        x=top20_mostUsage_task['description'],
        y=top20_mostUsage_task['job_card_no'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Mostly job done by plant for Retail Customers'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)
plot_dataOTH = [
    go.Scatter(
        x=top20_mostUsage_task_oth['description'],
        y=top20_mostUsage_task_oth['job_card_no'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Mostly job done by plant for Other Customers'
    )

fig = go.Figure(data=plot_dataOTH, layout=plot_layout)
iplot(fig)


# In[142]:


top20_expensive_task_OTH=df_transaction_Parts_others.groupby('description')['net_value'].mean().reset_index().sort_values(by='net_value',ascending=False).head(20)

top20_expensive_task=df_transaction_Parts_retail.groupby('description')['net_value'].mean().reset_index().sort_values(by='net_value',ascending=False).head(20)
plot_data = [
    go.Scatter(
        x=top20_expensive_task['description'],
        y=top20_expensive_task['net_value'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Expensive job done by plant for Retail customers'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)
plot_dataOTH = [
    go.Scatter(
        x=top20_expensive_task_OTH['description'],
        y=top20_expensive_task_OTH['net_value'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Expensive job done by plant for Other customers'
    )

fig = go.Figure(data=plot_dataOTH, layout=plot_layout)
iplot(fig)


# ### First observation from Job Time dataset
#      1.ACTION :DROP unnamed:_0  - no significant field 
#      2.ACTION :Keep deleting Item Category  "G2TX", 51629 observations have missing values in material,itemcatergory,netvalue.
#      3.ACTION: DROP labor value number having 67% missing values and not a significant field
# ##### Useful fields - 
#     1.DBM Order  - NO MISSING
#     2.Order_item - NO MISSING
#     3.Material   - NO MISSING
#     4.Labor Value Number- MISSING
#     5.Description-  NO MISSING
#     6.Item Catergory - NO MISSING
#     7.Order Quantity - NO MISSING
#     8.Target Quanity UOM - NO MISSING
#     9. NET Value - NO MISSING
# #### Loading Customer Dataset

# In[143]:


displayDataSetInfo(df_customer_data,'Customer Master')


# In[144]:


# drop missing value more than 60% 
df_customer_data.drop(columns={'title','customer_no','date_of_birth','marital_status','death_date','occupation'},axis=1,inplace=True)


# In[145]:


df_customer_data.info()


# In[146]:


# how many total missing values do we have?
total_cells = np.product(df_customer_data.shape)
missing_value_count=df_customer_data.isnull().sum()
total_missing = missing_value_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100


# #### Display Invoice Date

# In[147]:


displayDataSetInfo(df_transaction_data,'Invoice transaction')


# In[4]:


# no significant fields - GST
# drop missing value more than 60% 
df_transaction_data.drop(columns=['column1','service_advisor_name','claim_no','expiry_date',                                                                  'policy_no','gate_pass_date',                                                                     'user_id'],inplace=True)


# In[5]:


df_transaction_data.drop(columns=['area_/_locality','amt_rcvd_from_custom','amt_rcvd_from_ins_co'                                 ,'cash_/cashless_type','cgst14%', 'cgst25%', 'cgst6%', 'cgst9%'                                  ,'igst12%','igst18%', 'igst28%', 'igst5%'                                 ,'sgst/ugst14%', 'sgst/ugst25%', 'sgst/ugst6%', 'sgst/ugst9%'                                 ,'odn_no','tds_amount','outstanding_amt',                                 'total_cgst','total_gst', 'total_igst', 'total_sgst/ugst', 'total_value'],inplace=True)


# In[6]:


#insurance_company
df_transaction_data_copy=df_transaction_data.copy()
df_transaction_data.drop(columns=['regn_no'],inplace=True)
print(df_transaction_data.columns)
df_transaction_data.isnull().sum()


# In[48]:


df_transaction_data.query('insurance_company==insurance_company').to_csv("D:/Grayatom/Mahindra/file/CleanData/transaction_insured.csv")


# In[7]:


df_transaction_data.drop(columns=['insurance_company'],inplace=True)


# In[12]:


# City- fill from plant based on pincode
df_transaction_data.drop(columns=['city'],inplace=True)


# In[153]:


# how many total missing values do we have?
total_cells = np.product(df_transaction_data['technician_name'].shape)
missing_value_count=df_transaction_data['technician_name'].isnull().sum()
total_missing = missing_value_count.sum()

# percent of data that is missing
(total_missing/total_cells) * 100
# City,model,technician_name,regn_no- try to filling up- it was not captured


# In[52]:


df_transaction_data.query('technician_name==technician_name').to_csv("D:/Grayatom/Mahindra/file/CleanData/transaction_withtechnician.csv")


# In[8]:


df_transaction_data.drop(columns=['technician_name','model'],inplace=True)


# In[9]:


df_transaction_data.rename(columns={'pin_code':'postal_code'},inplace=True)
df_transaction_data.isnull().sum()


# ### Observation from Invoice data
# #### ACTION:DROP Technician Name,Model  , not significant feature for model.
# #### Give insight to business for techinician performance and Car model value in # of transactions
# #### Interactive Visualization

# In[156]:


df_plantmaster_data['state'].iplot(kind='hist', xTitle='State',x='state',y='plant_code',theme='white',
                  yTitle='Workshops', title='Total Workshops (State Wise)')
df_plantmaster_data['sales_organization'].iplot(kind='hist', xTitle='Sales Organization',x='sales_organization',y='state',
                theme='white',  yTitle='Total Workshops (count)', title='Total Workshops (Sales Organization)',legend=True)


# #### FInd Out Metric to measure business value for Mahindra

# In[10]:


#creating YearMonth field for the ease of reporting and visualization
df_transaction_data['InvoiceYearMonth'] = df_transaction_data['invoice_date'].map(lambda date: 100*date.year + date.month)
df_transaction_data_revenue = df_transaction_data.groupby(['InvoiceYearMonth'])['total_amt_wtd_tax'].sum().reset_index()
df_transaction_data_revenue.head(5)


# In[11]:


#X and Y axis inputs for Plotly graph. We use Scatter for line graphs
plot_data = [
    go.Scatter(
        x=df_transaction_data_revenue['InvoiceYearMonth'],
        y=df_transaction_data_revenue['total_amt_wtd_tax'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[12]:


import plotly.graph_objs as go
#using pct_change() function to see monthly percentage change
df_transaction_data_revenue['MonthlyGrowth'] = df_transaction_data_revenue['total_amt_wtd_tax'].pct_change()

#showing first 5 rows
df_transaction_data_revenue.head(5)

#visualization - line graph
plot_data = [
    go.Scatter(
        x=df_transaction_data_revenue.query("InvoiceYearMonth < 201612")['InvoiceYearMonth'],
        y=df_transaction_data_revenue.query("InvoiceYearMonth < 201612")['MonthlyGrowth'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Growth Rate'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[13]:


#creating a new dataframe with Retail customers only
df_transaction_data_retail = df_transaction_data.query("cust_type=='Retail'").reset_index(drop=True)

#creating monthly active customers dataframe by counting unique Customer IDs
retail_monthly_active = df_transaction_data_retail.groupby('InvoiceYearMonth')['customer_no'].nunique().reset_index()

#print the dataframe
retail_monthly_active.head(10)

#plotting the output
plot_data = [
    go.Scatter(
        x=retail_monthly_active['InvoiceYearMonth'],
        y=retail_monthly_active['customer_no'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Active Customers'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[14]:


#create a new dataframe for no. of order by using quantity field
retail_monthly_sales = df_transaction_data_retail.groupby('InvoiceYearMonth')['total_amt_wtd_tax'].sum().reset_index()

#print the dataframe
retail_monthly_sales

#plot
plot_data = [
    go.Scatter(
        x=retail_monthly_sales['InvoiceYearMonth'],
        y=retail_monthly_sales['total_amt_wtd_tax'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly order revenue by Retail'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[15]:


# create a new dataframe for average revenue by taking the mean of it
retail_monthly_order_avg = df_transaction_data_retail.groupby('InvoiceYearMonth')['total_amt_wtd_tax'].mean().reset_index()

#print the dataframe
retail_monthly_order_avg

#plot the bar chart
plot_data = [
    go.Scatter(
        x=retail_monthly_order_avg['InvoiceYearMonth'],
        y=retail_monthly_order_avg['total_amt_wtd_tax'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Order Average Revenue by Retail'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# New Customer Ratio: a good indicator of if we are losing our existing customers or unable to attract new ones
# 
# Retention Rate: King of the metrics. Indicates how many customers we retain over specific time window. 
# We will be showing examples for monthly retention rate and cohort based retention rate.

# In[171]:


df_transaction_data_retail.head(5)


# In[16]:


retail_min_purchase = df_transaction_data_retail.groupby('customer_no').invoice_date.min().reset_index()
retail_min_purchase.columns = ['customer_no','MinPurchaseDate']
retail_min_purchase['MinPurchaseYearMonth'] = retail_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)

df_transaction_data_retail = pd.merge(df_transaction_data_retail, retail_min_purchase, on='customer_no')


# In[17]:


#get the max purchase date for each customer and create a dataframe with it
retail_max_purchase = df_transaction_data_retail.groupby('customer_no').invoice_date.max().reset_index()
retail_max_purchase.columns = ['customer_no','MaxPurchaseDate']


# In[18]:


#create a column called User Type and assign Existing 
#if User's First Purchase Year Month before the selected Invoice Year Month
df_transaction_data_retail['UserType'] = 'New'
df_transaction_data_retail.loc[df_transaction_data_retail['InvoiceYearMonth']>df_transaction_data_retail['MinPurchaseYearMonth'],'UserType'] = 'Existing'
#retail_max_purchase.columns = ['customer_no','MaxPurchaseDate']
#calculate the Revenue per month for each user type
retail_user_type_revenue = df_transaction_data_retail.groupby(['InvoiceYearMonth','UserType'])['total_amt_wtd_tax'].sum().reset_index()

#filtering the dates and plot the result
retail_user_type_revenue = retail_user_type_revenue.query("InvoiceYearMonth != 201201 and InvoiceYearMonth != 201612")
plot_data = [
    go.Scatter(
        x=retail_user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],
        y=retail_user_type_revenue.query("UserType == 'Existing'")['total_amt_wtd_tax'],
        name = 'Existing'
    ),
    go.Scatter(
        x=retail_user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],
        y=retail_user_type_revenue.query("UserType == 'New'")['total_amt_wtd_tax'],
        name = 'New'
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New vs Existing'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[19]:


#create a dataframe that shows new user ratio - we also need to drop NA values (first month new user ratio is 0)
retail_new_user_ratio = df_transaction_data_retail.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['customer_no'].nunique()/df_transaction_data_retail.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['customer_no'].nunique() 
retail_new_user_ratio = retail_new_user_ratio.reset_index()
retail_new_user_ratio = retail_new_user_ratio.dropna()

#print the dafaframe
retail_new_user_ratio

#plot the result

plot_data = [
    go.Bar(
        x=retail_new_user_ratio.query("InvoiceYearMonth>201201 and InvoiceYearMonth<201612")['InvoiceYearMonth'],
        y=retail_new_user_ratio.query("InvoiceYearMonth>201201 and InvoiceYearMonth<201612")['customer_no'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New Customer Ratio'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[20]:


#identify which users are active by looking at their revenue per month
retail_user_purchase = df_transaction_data_retail.groupby(['customer_no','InvoiceYearMonth'])['total_amt_wtd_tax'].sum().reset_index()

#create retention matrix with crosstab
retail_user_retention = pd.crosstab(retail_user_purchase['customer_no'], retail_user_purchase['InvoiceYearMonth']).reset_index()

retail_user_retention.head(5)

#create an array of dictionary which keeps Retained & Total User count for each month
months = retail_user_retention.columns[2:]
retention_array = []
for i in range(len(months)-1):
    retention_data = {}
    selected_month = months[i+1]
    prev_month = months[i]
    retention_data['InvoiceYearMonth'] = int(selected_month)
    retention_data['TotalUserCount'] = retail_user_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = retail_user_retention[(retail_user_retention[selected_month]>0) & (retail_user_retention[prev_month]>0)][selected_month].sum()
    retention_array.append(retention_data)
    
#convert the array to dataframe and calculate Retention Rate
retail_user_retention = pd.DataFrame(retention_array)
retail_user_retention['RetentionRate'] = retail_user_retention['RetainedUserCount']/retail_user_retention['TotalUserCount']

#plot the retention rate graph
plot_data = [
    go.Scatter(
        x=retail_user_retention.query("InvoiceYearMonth<201612")['InvoiceYearMonth'],
        y=retail_user_retention.query("InvoiceYearMonth<201612")['RetentionRate'],
        name="organic"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Retention Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# ### Customer Segmentation
# #### Cluster algorithm - K-means 
# #### Recency,Frequency,Monetory Clusters
# #### Create Customer segment using Overall Score.

# In[21]:


#create a generic user dataframe to keep CustomerID and new segmentation scores
retail_user = pd.DataFrame(df_transaction_data_retail['customer_no'].unique())
retail_user.columns = ['customer_no']


#we take our observation point as the max invoice date in our dataset
retail_max_purchase['Recency'] = (retail_max_purchase['MaxPurchaseDate'].max() - retail_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
retail_user = pd.merge(retail_user, retail_max_purchase[['customer_no','Recency']], on='customer_no')

retail_user.head(5)

#plot a recency histogram

plot_data = [
    go.Histogram(
        x=retail_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[181]:


retail_user.Recency.describe()


# In[22]:


sse={}
tx_recency = retail_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[23]:


#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(retail_user[['Recency']])
retail_user['RecencyCluster'] = kmeans.predict(retail_user[['Recency']])


retail_user = order_cluster('RecencyCluster', 'Recency',retail_user,False)


# In[184]:


retail_user.groupby('RecencyCluster').describe()


# In[24]:


#get order counts for each user and create a dataframe with it
retail_user_frequency = df_transaction_data_retail.groupby('customer_no').invoice_date.count().reset_index()
retail_user_frequency.columns = ['customer_no','Frequency']

#add this data to our main dataframe
retail_user = pd.merge(retail_user, retail_user_frequency, on='customer_no')

#plot the histogram
plot_data = [
    go.Histogram(
        x=retail_user.query('Frequency < 1000')['Frequency']
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[25]:


#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(retail_user[['Frequency']])
retail_user['FrequencyCluster'] = kmeans.predict(retail_user[['Frequency']])

#order the frequency cluster
retail_user = order_cluster('FrequencyCluster', 'Frequency',retail_user,True)

#see details of each cluster
retail_user.groupby('FrequencyCluster')['Frequency'].describe()


# In[26]:


#get order counts for each user and create a dataframe with it
retail_user_monetory = df_transaction_data_retail.groupby('customer_no').total_amt_wtd_tax.sum().reset_index()
retail_user_monetory.columns = ['customer_no','Monetary']

#add this data to our main dataframe
retail_user = pd.merge(retail_user, retail_user_monetory, on='customer_no')

#plot the histogram
plot_data = [
    go.Histogram(
        x=retail_user.query('Monetary < 10000')['Monetary']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[194]:


retail_user.head(5)


# In[27]:


#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(retail_user[['Monetary']])
retail_user['RevenueCluster'] = kmeans.predict(retail_user[['Monetary']])


#order the cluster numbers
retail_user = order_cluster('RevenueCluster', 'Monetary',retail_user,True)

#show details of the dataframe
retail_user.groupby('RevenueCluster')['Monetary'].describe()


# In[28]:


#calculate overall score and use mean() to see details
retail_user['OverallScore'] = retail_user['RecencyCluster'] + retail_user['FrequencyCluster'] + retail_user['RevenueCluster']
retail_user[['Recency','Frequency','Monetary','OverallScore']].groupby('OverallScore').mean()


# In[29]:


retail_user['Segment'] = 'Low-Value'
retail_user.loc[retail_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
retail_user.loc[retail_user['OverallScore']>4,'Segment'] = 'High-Value' 


# In[200]:


print(retail_user.shape)
retail_user.head(5)


# High Value: Improve Retention
# Mid Value: Improve Retention + Increase Frequency
# Low Value: Increase Frequency

# ### Calculating Customer Life Time Value
# #### Who are our best one's customer by measuring LTV metric
# #### Lifetime Value: Total Gross Revenue - Total Cost
# ##### This equation now gives us the historical lifetime value. If we see some customers having very high negative lifetime value historically, it could be too late to take an action. At this point, we need to predict the future with machine learning:
# 
# ### Lifetime Value Prediction
# ##### Define an appropriate time frame for Customer Lifetime Value calculation
# ##### Identify the features we are going to use to predict future and create them
# ##### Calculate lifetime value (LTV) for training the machine learning model
# ##### Build and run the machine learning model
# ##### Check if the model is useful

# In[30]:


retail_graph = retail_user.query("Monetary < 50000 and Frequency < 2000")
df_transaction_data_retail.to_csv("D:/Grayatom/Mahindra/file/CleanData/retail_RFM_data.csv")


# In[3]:


df_transaction_data_retail=pd.read_csv("D:/Grayatom/Mahindra/file/CleanData/retail_RFM_data.csv")


# In[4]:


df_transaction_data_retail[["invoice_date", "jobcard_date"]] = df_transaction_data_retail[["invoice_date", "jobcard_date"]].apply(pd.to_datetime)


# In[5]:


df_transaction_data_retail.isnull().sum()


# In[6]:


#read data from csv and redo the data work we done before
# Define two dataframe for 3 months and 6 months
# predict 6 month data based on 3 month 
#create 3m and 1m dataframes
import datetime
tx_3m = df_transaction_data_retail[(df_transaction_data_retail.invoice_date < datetime.date(2016,6,1)) &                                    (df_transaction_data_retail.invoice_date >= datetime.date(2016,3,1))].reset_index(drop=True)
tx_6m = df_transaction_data_retail[(df_transaction_data_retail.invoice_date >= datetime.date(2016,6,1)) &                                    (df_transaction_data_retail.invoice_date < datetime.date(2016,12,1))].reset_index(drop=True)

#create tx_user for assigning clustering
tx_user = pd.DataFrame(tx_3m['customer_no'].unique())
tx_user.columns = ['CustomerID']


#calculate recency score
tx_max_purchase = tx_3m.groupby('customer_no').invoice_date.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

#calcuate frequency score
tx_frequency = tx_3m.groupby('customer_no').invoice_date.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']
tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

#calcuate revenue score
tx_3m['Revenue'] = tx_3m['total_amt_wtd_tax']
tx_revenue = tx_3m.groupby('customer_no').Revenue.sum().reset_index()
tx_revenue.columns = ['CustomerID','Revenue']
tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)


#overall scoring
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 


# In[7]:


## Create RFM scoring dataframe
tx_user.head(5)


# #### Since our feature set is ready, let’s calculate 6 months LTV for each customer which we are going to use for training our model.
# #### There is no cost specified in the dataset. That’s why Revenue becomes our LTV directly.

# In[9]:


#calculate revenue and create a new dataframe for it
tx_6m['Revenue'] = tx_6m['total_amt_wtd_tax']
tx_user_6m = tx_6m.groupby('customer_no')['Revenue'].sum().reset_index()
tx_user_6m.columns = ['CustomerID','m6_Revenue']


#plot LTV histogram
plot_data = [
    go.Histogram(
        x=tx_user_6m.query('m6_Revenue < 10000')['m6_Revenue']
    )
]

plot_layout = go.Layout(
        title='6m Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# In[10]:


tx_merge = pd.merge(tx_user, tx_user_6m, on='CustomerID', how='left')
tx_merge = tx_merge.fillna(0)

tx_graph = tx_merge.query("m6_Revenue < 30000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Low-Value'")['m6_Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Mid-Value'")['m6_Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'High-Value'")['m6_Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "6m LTV"},
        xaxis= {'title': "RFM Score"},
        title='LTV'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
iplot(fig)


# ### Positive correlation is quite visible here. High RFM score means high LTV.
# 
# Before building the machine learning model, we need to identify what is the type of this machine learning problem. LTV itself is a regression problem. A machine learning model can predict the $ value of the LTV. But here, we want LTV segments. Because it makes it more actionable and easy to communicate with other people. By applying K-means clustering, we can identify our existing LTV groups and build segments on top of it.
# Considering business part of this analysis, we need to treat customers differently based on their predicted LTV. For this example, we will apply clustering and have 3 segments (number of segments really depends on your business dynamics and goals):
# 
# #### Low LTV
# #### Mid LTV
# #### High LTV

# In[11]:


#remove outliers
tx_merge = tx_merge[tx_merge['m6_Revenue']<tx_merge['m6_Revenue'].quantile(0.99)]


#creating 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_merge[['m6_Revenue']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['m6_Revenue']])

#order cluster number based on LTV
tx_merge = order_cluster('LTVCluster', 'm6_Revenue',tx_merge,True)

#creatinga new cluster dataframe
tx_cluster = tx_merge.copy()

#see details of the clusters
tx_cluster.groupby('LTVCluster')['m6_Revenue'].describe()


# In[12]:


def segment_to_numeric(x):
    if x=='Low-Value':
        return 0
    if x=='Mid-Value':
        return 1
    if x=='High-Value':
        return 2
    


# In[12]:


def segment_to_numeric(x):
    if x=='Low-Value':
        return 0
    if x=='Mid-Value':
        return 1
    if x=='High-Value':
        return 2
    


# #### 2 is the best with average 24.8k LTV whereas 0 is the worst with 307.

# In[368]:


#convert categorical columns to numerical
tx_cluster['segment_num'] = tx_cluster.Segment.apply(segment_to_numeric)
tx_class=tx_cluster.copy()
tx_class.drop(columns=['Segment'],inplace=True)

#create X and y, X will be feature set and y is the label - LTV
X = tx_class.drop(['LTVCluster','m6_Revenue'],axis=1)
y = tx_class['LTVCluster']
from sklearn.model_selection import KFold, cross_val_score, train_test_split
#split training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[369]:


#calculate and show correlations
corr_matrix = tx_class.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)


# In[370]:


X_train.drop(columns=['CustomerID'],inplace=True)
X_test.drop(columns=['CustomerID'],inplace=True)


# In[207]:


from collections import Counter

a = dict(Counter(y_train))
print(a)
print(y_train.shape[0])
print("Ratio for LTV Cluster 0", a[0]/y_train.shape[0])
print("Ratio for LTV Cluster 2", a[1]/y_train.shape[0])
print("Ratio for LTV Cluster 1", a[2]/y_train.shape[0])


# In[371]:


from sklearn.feature_selection import f_classif,chi2,SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2

# # f_classif
skb = SelectPercentile(f_classif, percentile=95)
predictors=skb.fit_transform(X_train, y_train)
scores=skb.scores_.tolist() 
print(scores)
Features=X_train.columns
print(Features)
scaledfeatured_df=pd.DataFrame({'features':Features,'scores':scores})
scaledfeatured_df.sort_values(by='scores',ascending=None,inplace=True)
top_fclassif_predictors = list(scaledfeatured_df['features'][:predictors.shape[1]])
top_fclassif_predictors


# ### Build Model XGB for multilabel class problem.

# In[372]:


from sklearn.metrics import classification_report
import xgboost as xgb
#XGBoost Multiclassification Model
ltv_xgb_model = xgb.XGBClassifier(max_depth=3, learning_rate=0.0025,n_jobs=-1).fit(X_train[top_fclassif_predictors], y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(ltv_xgb_model.score(X_train[top_fclassif_predictors], y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(ltv_xgb_model.score(X_test[top_fclassif_predictors], y_test)))

y_pred_xgb = ltv_xgb_model.predict(X_test[top_fclassif_predictors])
print(classification_report(y_test, y_pred_xgb))
plt.scatter(y_test, y_pred_xgb)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# In[373]:


print(confusion_matrix(y_test, y_pred_xgb))


# ### Precision and recall are acceptable for 0. As an example, for cluster 0 (Low LTV), if model tells us this customer belongs to cluster 0, 90 out of 100 will be correct (precision). And the model successfully identifies 100% of actual cluster 0 customers (recall). We really need to improve here because our model is going to overfit.
# 
# 
# ### And we need to check ffor the model for other clusters. 
# 
# ### For example, we did not detect Mid and high LTV customers. Possible actions to improve those points:
# 
# #### Adding more features and improve feature engineering
# #### Try different models other than XGBoost
# #### Apply hyper parameter tuning to current model
# #### Add more data to the model if possible

# In[298]:


#imbalance Dataset
from imblearn.over_sampling import SMOTE,RandomOverSampler
from collections import Counter
sm = RandomOverSampler(random_state=42)
X_train_sample, y_train_sample = sm.fit_sample(X_train, y_train)
X_train_sample.shape,X_train.shape


# In[215]:



a = dict(Counter(y_train_sample))
print(a)
print(y_train_sample.shape[0])
print("Ratio for LTV Cluster 0", a[0]/y_train_sample.shape[0])
print("Ratio for LTV Cluster 2", a[1]/y_train_sample.shape[0])
print("Ratio for LTV Cluster 1", a[2]/y_train_sample.shape[0])


# ## Try with Multiple Classifier

# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
### Set Transformers for pipeline
pipe = Pipeline(steps=[('scale', StandardScaler()), 
                       ('classifier',(LogisticRegression()))])
pipe.steps
clfs=[LogisticRegression(),
         DecisionTreeClassifier(),
         GradientBoostingClassifier(),
         RandomForestClassifier(),
         KNeighborsClassifier(),
     xgb.XGBClassifier()]

cross_validate_with_multiclassifier(clfs,pipe,X_train_sample,y_train_sample)


# In[374]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
### Set Transformers for pipeline
from sklearn.metrics import classification_report
import xgboost as xgb

pipe = Pipeline(steps=[('scale', StandardScaler()), 
                       ('classifier',(LogisticRegression()))])
pipe.steps
clfs=[LogisticRegression(),
     xgb.XGBClassifier(max_depth=3, learning_rate=0.05).fit(X_train_sample[top_fclassif_predictors], y_train_sample)]

cross_validate_with_multiclassifier(clfs,pipe,X_train_sample,y_train_sample)


# ### Hyper Paramter Tuning using Ensemble technique for multiclass classification
# #### Our goal is to predict High and mid value customer correctly by model.

# In[386]:


parameter_grid = {"classifier__learning_rate":[0.0017],
              "classifier__max_depth": [3],
              "classifier__booster": ["gbtree"],
              "classifier__n_estimators":[150], 
                "classifier__objective":["multi:softprob"],
                 "classifier__num_class":[3],
                  "classifier__gamma":[0]}
from sklearn.metrics import accuracy_score
#pipe.set_params(classifier=DecisionTreeClassifier())
#pipe.set_params(classifier=SVC())
pipe.set_params(classifier=xgb.XGBClassifier())
cv_grid = GridSearchCV(pipe, param_grid = parameter_grid)

cv_grid.fit(X_train_sample, y_train_sample)
print("best combination of the parameters -", cv_grid.best_params_)
print("best estimator -",cv_grid.best_estimator_)
print("best score -",cv_grid.best_score_)
y_pred = cv_grid.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy of the best classifier after CV is %.3f%%' % (accuracy*100))


# In[387]:


from sklearn.metrics import f1_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#best score - 0.9361993820368758
#Accuracy of the best classifier after CV is 78.186%
print("Evaluation ")

precision_score_xgb_micro=precision_score(y_test,y_pred,average='micro')
precision_score_xgb_macro=precision_score(y_test,y_pred,average='macro')
recall_score_xgb_micro=recall_score(y_test,y_pred,average='micro')
recall_score_xgb_macro=recall_score(y_test,y_pred,average='macro')

f1_score_xgb_micro=f1_score(y_test,y_pred,average='micro')
f1_score_xgb_macro=f1_score(y_test,y_pred,average='macro')
f1_score_xgb_weigthage=f1_score(y_test,y_pred,average='weighted')
print("average micro score for precision -",precision_score_xgb_micro)
print("average macro score for precision -",precision_score_xgb_macro)
print("average micro score for recall -",recall_score_xgb_micro)
print("average macro score for recall -",recall_score_xgb_macro)
print("average micro score for f1 -",f1_score_xgb_micro)
print("average macro score for f1 -",f1_score_xgb_macro)
print("average weighted score for f1 -",f1_score_xgb_weigthage)


# In[345]:


from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
# generate the precision recall curve
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn import svm
# Use label_binarize to be multi-label like settings
Y = label_binarize(y_train_sample, classes=[0, 1, 2])
n_classes = Y.shape[1]

# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X_train_sample, Y, test_size=.2, random_state=42)

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

clf_SVM = OneVsRestClassifier(LinearSVC())
params = {
      'estimator__C': [1]}
gs = GridSearchCV(clf_SVM, params, cv=5)
gs.fit(X_train, Y_train)


# In[346]:


y_pred_SVC=gs.predict(X_test)
y_score = gs.decision_function(X_test)


# In[347]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


# In[348]:


plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))


# In[349]:


from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.show()


# In[350]:


from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
precision["macro"], recall["macro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["macro"] = average_precision_score(Y_test, y_score,
                                                     average="macro")
print('Average precision score, macro-averaged over all classes: {0:0.2f}'
      .format(average_precision["macro"]))

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["macro"], precision["macro"], color='gold', lw=2)
lines.append(l)
labels.append('macro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["macro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.show()


# In[351]:


#Accuracy of the best classifier after CV is 78.186%
def evaluatePerf(true_labels, predicted_labels):
    type2 = 0
    type1 = 0
    true_positive = 0
    true_negative = 0
    for x,y in zip(true_labels, predicted_labels):
        if x == y:
            if x == 1:
                true_positive+=1
            else:
                true_negative+=1
        elif x == 1:
            type2 += 1
        elif x == 0:
            type1 += 1

    print("TP:", true_positive, " TN:", true_negative, " T1Err:", type1, " T2Err:", type2)
    


# 

# 

# ### Observation
# ### we have discovered tracking essential metrics, customer segmentation, and predicting the lifetime value programmatically
# 
# #### SVM classifier is able to predict small classes as well. We can see micro average score  - 40% and macro average score - 39% for this model.
# 
# #### As per evaluating and problem, XGB Classfier is the our best model to predict each class - LTV cluster 0,1 and 2.
# #### Average micro score for f1 - 0.6192849134172169
# #### Average macro score for f1 - 0.34212866400473313
# #### Average weighted score for f1 - 0.7046729549064478
# 
# 
# #### Precision and Recall are now acceptable for LTV Cluster -1 (MID VALUE) and 2 (High Value) also as we had low ratio of those customers and our model is able to achive our business goal.
# #### Our model is 65% detecting Low value customer ,36% Mid value and 30% high value customer.
# 
# #### Now our model is able to identify small classes also. 
# #### Due to imbalanced dataset, we use f1 weighted score which tells us the average precision and recall score. 
# ### Our model achieve 70% f1-score.

# # We are able to segment the customers -
#     Low Value
#     Mid Value
#     High Value
#     
# # XGB Classifier is a good model to predict each class equally in imbalance dataset with 70% average f1 score.
#     
# # We can start taking actions with this segmentation. The main strategies are quite clear:
# High Value: Improve Retention
# Mid Value: Improve Retention + Increase Frequency
# Low Value: Increase Frequency
