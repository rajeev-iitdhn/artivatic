#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install textblob


# In[15]:


pip install xgboost


# In[65]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob


# In[66]:


path="/home/dell/Desktop/artivatic/ML_Artivatic_dataset"

train_data=path+"/train_indessa.csv"
val_data=path+"/test_indessa.csv"
sample_submission=path+"/sample_submission.csv"

train_data=pd.read_csv(train_data)
val_data=pd.read_csv(val_data)
sample_submission=pd.read_csv(sample_submission)


# In[67]:


copied=train_data.copy()


# In[68]:


le = LabelEncoder()
def batch_enroll(x):
    if x==" ":
        return 0
    else:
        return len(str(x))
def string_replacer(x):
    x=str(x).replace("<","")
    x=str(x).replace("+","")
    return x
def string_strip(x):
    x=str(x).replace("<","")
    x=str(x).replace("+","")
    if "years" in x:
        x=x.strip("years")
    if "year" in x:
        x=x.strip("year")
    if "months" in x:
        x=x.strip("months")
    if "th week" in x:
        x=x.strip("th week")
    return x
def sentiment(x):
    blob=TextBlob(str(x))
    result=round(blob.sentiment[0],2)
    return result
def interpolat(x):
    x=x.interpolate()
    x=x.fillna(method="ffill")
    x=x.fillna(method="backfill")
    return x


# In[69]:


def col_pre(copied):
    input_feat=["loan_amnt","funded_amnt","funded_amnt_inv","batch_enrolled","int_rate",
             "grade","sub_grade","emp_title","purpose","addr_state","dti","delinq_2yrs","inq_last_6mths",
            "mths_since_last_delinq","mths_since_last_record","open_acc","pub_rec","revol_bal","revol_util",
            "total_acc","initial_list_status","total_rec_int","total_rec_late_fee","recoveries",
             "collection_recovery_fee","collections_12_mths_ex_med"
             ,"application_type",
            "acc_now_delinq","tot_coll_amt","tot_cur_bal","total_rev_hi_lim"]
    output_feat="loan_status"
    
    le_col=["grade","sub_grade","home_ownership","verification_status","pymnt_plan","purpose",
           "addr_state","initial_list_status","application_type"]
    for col in le_col:
        copied[col]=pd.Series(le.fit_transform(copied[col])).apply(int)
    copied["emp_title"]=pd.Series(le.fit_transform(copied.emp_title.apply(string_replacer)))
#     copied.verification_status_joint=le.fit_transform(copied.verification_status_joint.apply(str))

    copied.batch_enrolled=copied.batch_enrolled.apply(batch_enroll)
#     copied.emp_length=copied.emp_length.apply(string_strip)
    copied.term=copied.term.apply(string_strip)
    copied.last_week_pay=copied.last_week_pay.apply(string_strip)
    copied=copied[input_feat]
    copied=interpolat(copied)
    return copied


# In[ ]:





# In[70]:


input_feat=["loan_amnt","funded_amnt","funded_amnt_inv","batch_enrolled","int_rate",
             "grade","sub_grade","emp_title","purpose","addr_state","dti","delinq_2yrs","inq_last_6mths",
            "mths_since_last_delinq","mths_since_last_record","open_acc","pub_rec","revol_bal","revol_util",
            "total_acc","initial_list_status","total_rec_int","total_rec_late_fee","recoveries",
             "collection_recovery_fee","collections_12_mths_ex_med"
             ,"application_type",
            "acc_now_delinq","tot_coll_amt","tot_cur_bal","total_rev_hi_lim"]


# In[71]:


data=col_pre(train_data)


# In[95]:


test_=col_pre(val_data)


# In[73]:


data["loan_staus"]=train_data.loan_status


# In[75]:


train=data.iloc[:int(98*len(data)/100),:]
val=data.iloc[int(98*len(data)/100):,:]


# In[76]:


x_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]


# In[77]:


x_val=val.iloc[:,:-1]
y_val=val.iloc[:,-1]


# In[78]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_auc_score


# In[79]:


from xgboost import XGBClassifier


# In[91]:


mdl=xgb_mdl.fit(x_train,y_train)


# In[92]:


probs=mdl.predict_proba(x_val)


# In[93]:


probs = probs[:, 1]


# In[94]:


auc = roc_auc_score(y_val, probs)
print('AUC: %.2f' % auc)


# In[17]:


prediction=mdl.predict(x_val)


# In[96]:


output=mdl.predict(test_)


# In[101]:


test_["member_id"]=val_data.member_id


# In[110]:


output_df=pd.DataFrame()


# In[111]:


output_df["member_id"]=test_["member_id"]
output_df["loan_status"]=pd.Series(output)


# In[115]:


with pd.ExcelWriter('sample_submission.xlsx') as writer:  
    output_df.to_excel(writer, sheet_name='Sample_submission')

