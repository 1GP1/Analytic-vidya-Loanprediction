#!/usr/bin/env python
# coding: utf-8

# #### Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
# #### Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers. 

# In[1]:


# importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


df=pd.read_csv('train_ctrUa4K.csv')


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# # Exploratory Data Analysis
# 

# ## Applicant income

# In[8]:


sns.distplot(df.ApplicantIncome)


# In[9]:


sns.swarmplot(data=df,x='Loan_Status',y='ApplicantIncome')


# ## CoApplicant Income

# In[10]:


sns.distplot(df.CoapplicantIncome,bins=50)


# In[11]:


sns.swarmplot(data=df,x='Loan_Status',y='CoapplicantIncome')


# In[12]:


sns.pairplot(df.drop(['Credit_History','Loan_Amount_Term'],axis=1))


# In[13]:


sns.pairplot(df.drop(['Credit_History','Loan_Amount_Term'],axis=1),hue='Loan_Status')


# ## Gender

# In[14]:


df.groupby(['Gender','Loan_Status'])['Gender'].count()


# In[15]:


sns.countplot(df.Gender)


# In[16]:


# stacked bar plot male and female with loan status 
yes=df['Gender'][df['Loan_Status']=='Y'].value_counts()
no=df['Gender'][df['Loan_Status']=='N'].value_counts()
plt.bar(yes.index,yess)
plt.bar(no.index,no,bottom=yes)
plt.legend(labels=['yes','no'])


# In[17]:


a=df['Gender'][df['Loan_Status']=='Y'].dropna().value_counts().sort_index()/df.Gender.value_counts().sort_index()
plt.bar(a.sort_values(ascending=False).index,a.sort_values(ascending=False))
plt.title('Loan approval rate Vs Gender')
# Male and Female roughly have the same percentage of loan being approved


# ## Married

# In[18]:


sns.countplot(df['Married'])


# In[19]:


# stacked bar plot married and unmarried applicant with loan status
myes=df['Married'][df['Loan_Status']=='Y'].value_counts()
mno=df['Married'][df['Loan_Status']=='N'].value_counts()
plt.bar(myes.index,myes)
plt.bar(mno.index,mno,bottom=myes)
plt.legend(labels=['loan_approved','not approved'])
plt.title('Married')


# In[20]:


a=df['Married'][df['Loan_Status']=='Y'].dropna().value_counts().sort_index()/df.Married.value_counts().sort_index()
plt.bar(a.sort_values(ascending=False).index,a.sort_values(ascending=False))
plt.title('Loan approval rate Vs Married')
# married people have a higher chance of getting the loan approved


# In[102]:


sns.boxplot('Married','LoanAmount',data=df)


# In[104]:


sns.swarmplot('Married','ApplicantIncome',data=df)


# In[ ]:





# In[ ]:





# ##  Education

# In[ ]:





# In[23]:


# stacked bar plot Graduate and non graduate applicant with loan status
myes=df['Education'][df['Loan_Status']=='Y'].value_counts()
mno=df['Education'][df['Loan_Status']=='N'].value_counts()
plt.bar(myes.index,myes)
plt.bar(mno.index,mno,bottom=myes)
plt.legend(labels=['loan_approved','not approved'])
plt.title('Education')


# In[24]:


a=df['Education'][df['Loan_Status']=='Y'].dropna().value_counts().sort_index()/df.Education.value_counts().sort_index()
plt.bar(a.sort_values(ascending=False).index,a.sort_values(ascending=False))
plt.title('Loan approval rate Vs Education')
# Graduates have a higher chance of getting the loan approved


# In[25]:


sns.swarmplot('Education','ApplicantIncome',data=df)


# In[26]:


# Distribution of graduates and non-graduates among male and female
gend=pd.crosstab(df.Gender,df.Education)
cmap=plt.get_cmap('tab20c')
outer=cmap(np.array([0,4]))
inner=cmap(np.array([1,2,5,6]))
gend=pd.crosstab(df.Gender,df.Education)
plt.pie(gend.sum(axis=1),radius=1,wedgeprops=dict(width=0.3),colors=outer,labels=gend.index)
plt.pie(gend.values.flatten(),radius=0.7,wedgeprops=dict(width=0.3),colors=inner,labels=['G','NG','G','NG'],labeldistance=0.65)
plt.show()


# ## Property_area

# In[27]:


# stacked bar plot of property area distributed on loan status 
myes=df['Property_Area'][df['Loan_Status']=='Y'].value_counts().sort_index()
mno=df['Property_Area'][df['Loan_Status']=='N'].value_counts().sort_index()
plt.bar(myes.index,myes)
plt.bar(mno.index,mno,bottom=myes)
plt.legend(labels=['loan_approved','not approved'])
plt.title('Property_Area')


# In[28]:


a=df['Property_Area'][df['Loan_Status']=='Y'].dropna().value_counts().sort_index()/df.Property_Area.value_counts().sort_index()
plt.bar(a.sort_values(ascending=False).index,a.sort_values(ascending=False))
plt.title('Loan approval rate Vs Property_Area')
# There is a higher probability of loan approval for those who belongs to semiurban area
# semiurban shows largest loan approval rates


# In[29]:


#The percentage of loan approved gender wise in each property area
A=df.groupby(['Gender','Property_Area','Loan_Status']).count()/df.groupby(['Gender','Property_Area']).count()
A['Loan_ID']
# We can see that in semi-urban region both male and femal have much higher percentage of approval.


# In[108]:


sns.swarmplot('Property_Area','ApplicantIncome',data=df)


# In[ ]:





# ## loan amount term

# In[31]:


df.Loan_Amount_Term.value_counts()


# In[32]:


sns.countplot(df.Loan_Amount_Term)
# loan amount term is highly skewed data with majority of the loan term as 360 months.


# In[33]:


df.groupby('Loan_Amount_Term').mean()


# In[34]:


sns.kdeplot(df['LoanAmount'][df['Loan_Amount_Term']==360],shade=True)
sns.kdeplot(df['LoanAmount'][df['Loan_Amount_Term']==180],shade=True)
sns.kdeplot(df['LoanAmount'][df['Loan_Amount_Term']==300],shade=True)
sns.kdeplot(df['LoanAmount'][df['Loan_Amount_Term']==84],shade=True)
plt.legend(labels=['360','180','300','84'])
#there is no clear difference with distributiion of the loan amount and it can be divided into three classes
# long term , medium term and short term


# In[ ]:





# ## Dependents

# In[35]:


myes=df['Dependents'][df['Loan_Status']=='Y'].dropna().value_counts().sort_index()
mno=df['Dependents'][df['Loan_Status']=='N'].dropna().value_counts().sort_index()
plt.bar(myes.index,myes)
plt.bar(mno.index,mno,bottom=myes)
plt.legend(labels=['loan_approved','not approved'])
plt.title('Dependents')
# majority of the applicants have zero dependents, other classes have comparetivley much lower number.


# In[36]:


a=df['Dependents'][df['Loan_Status']=='Y'].dropna().value_counts().sort_index()/df.Dependents.value_counts()
plt.bar(a.sort_values(ascending=False).index,a.sort_values(ascending=False))
plt.title('Loan approval rate Vs Dependents')
# The percentage of loan approved is roughly the same for all applicant with diffrent number of dependents
# Applicants with 2 dependents shows slightly higher percentage of approval rate 


# In[37]:


sns.swarmplot('Dependents','ApplicantIncome',data=df)


# ## self employed

# In[38]:


sns.countplot(df.Self_Employed)


# In[39]:


myes=df['Self_Employed'][df['Loan_Status']=='Y'].dropna().value_counts().sort_index()
mno=df['Self_Employed'][df['Loan_Status']=='N'].dropna().value_counts().sort_index()
plt.bar(myes.index,myes)
plt.bar(mno.index,mno,bottom=myes)
plt.legend(labels=['loan_approved','not approved'])
plt.title('Self_Employed')
plt.show()
# Majority of the people who apply for loan are not self employed ie. they have a regular source of income


# In[40]:


a=df['Self_Employed'][df['Loan_Status']=='Y'].dropna().value_counts()/df.Self_Employed.value_counts()
plt.bar(a.index,a)
plt.title('Loan approval rate Vs Self_Employed')
plt.show()
# Eventhough self employed is higher in number, the percentage of people who's loan is approved seems to be the same.


# In[41]:


sns.swarmplot('Self_Employed','ApplicantIncome',data=df)


# # credit history

# In[42]:


# credit history gives an idea about how the applicant had paid of credit previously
myes=df['Credit_History'][df['Loan_Status']=='Y'].dropna().value_counts().sort_index()
mno=df['Credit_History'][df['Loan_Status']=='N'].dropna().value_counts().sort_index()
plt.bar(myes.index,myes)
plt.bar(mno.index,mno,bottom=myes)
plt.legend(labels=['loan_approved','not approved'])
plt.title('Credit_History')
plt.show()
# It can be noticed that credit history have a direct relationship with the loan_status
# majority of the applicant with credit history 1 have their loan approved
# majority of the applicant with credit history 0 have their loan denied


# In[43]:


sns.swarmplot(x='Credit_History',y='ApplicantIncome',data=df,hue='Loan_Status')
plt.title('credit_history Vs Income')


# In[44]:


sns.swarmplot(x='Loan_Status',y='ApplicantIncome',data=df[df.Credit_History.isnull()])


# # Creating a pipeline for data pre processing

# In[45]:


# number of null values
df.isnull().sum()


# In[46]:


# function to divide the loan amount tern to three classes 
# long , medium and short term
def loan_term(a):
    if a>=300:
        return 'long_term'
    if (a>=120) & (a<300):
        return 'medium_term'
    if (a<120):
        return 'short_term'


# In[122]:


from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[123]:


# numerical_1 and categorical_1 are attributes present pre data processing
# numerical_2 and categorical_2 will be used to add new features and contains the column name post data processing
global numerical_1
numerical_1=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       ]
global categorical_1
categorical_1=['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Credit_History', 'Property_Area','Loan_Amount_Term']
global column
column=numerical_1+categorical_1

column_2=column.copy()  
numerical_2=numerical_1.copy()
categorical_2=categorical_1.copy()


# In[138]:


# pipeline has been divided into 3 components
# 1. imputer for categorical and numerical data to fill missing values
# 2. customized transformer for feature engineering 
# 3. standard scaler for numerical data and one hot encoder for categorical data
imputer=ColumnTransformer([
    ('num',SimpleImputer(strategy='mean'),numerical_1),
    ('cat',SimpleImputer(strategy='most_frequent'),categorical_1)
])
# customizd transformer for feature engineering and cleaning data
class feature_eng(BaseEstimator,TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        global column_2
        global numerical_2
        global categorical_2
        column_2=column.copy()
        numerical_2=numerical_1.copy()
        categorical_2=categorical_1.copy()
        x=pd.DataFrame(X,columns=column_2)
        # creating a new attribute total income ie. sum of applicant and co applicant income.
        total_income=x['ApplicantIncome']+x['CoapplicantIncome']
        x.insert(len(numerical_2),'total_income',total_income)
        numerical_2.append('total_income')
        # Debt per income gives us an understanding of the financial pressure on the applicant 
        debt_per_income=x['LoanAmount']/x['total_income']
        x.insert(len(numerical_2),'debt_per_income',debt_per_income)
        numerical_2.append('debt_per_income')
        
        x['Loan_Amount_Term']=x['Loan_Amount_Term'].apply(loan_term)
        x['Married']=x['Married'].apply(lambda x:'Married' if x=='Yes' else 'not_married')
        x['Self_Employed']=x['Self_Employed'].apply(lambda x:'Self_Employed' if x=='Yes' else 'Not_Self_Employed')
        
        column_2=numerical_2+categorical_2

        
        return x
      
encoder=ColumnTransformer([
    ('num',StandardScaler(),numerical_2),
    ('cat',OneHotEncoder(),categorical_2)
])    


# In[139]:


# final pipeline
pre_processing=Pipeline([
    ('imputer',imputer),
    ('feature_eng',feature_eng()),
    ('encoder',encoder)
])


# In[140]:


def preprocessing(df,target=True):
    # function that passes the dataframe through a pipeline and returns the final clean data and column names
    X=pre_processing.fit_transform(df)
    cat_encoded=pre_processing.named_steps['encoder'].named_transformers_['cat']
    cat_one_hot=[j for i in cat_encoded.categories_ for j in i ]
    column_post_processing=numerical_2+cat_one_hot

    if target:
        # when the target variable is present in the data frame passed to the function
        y=df.Loan_Status.apply(lambda x : 1 if x=='Y' else 0).values
        return X ,y ,column_post_processing
    else:
        return X ,column_post_processing


# # Train test split

# In[141]:


train=pd.read_csv('train_ctrUa4K.csv')


# In[142]:


sns.countplot(x='Loan_Status',data=df)
#As the number of positive cases are much higher than the negative cases
# we need to make sure that both train and test case has equal propotion of both classes.
#This can be achieved by doing a strattified shuffle split


# In[143]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[144]:


X,y,columns_post=preprocessing(train)


# In[148]:


len(columns_post)


# In[147]:


X.shape


# In[149]:


split=StratifiedShuffleSplit(n_splits=1,test_size=0.3)


# In[150]:


split.split(X,y)


# In[151]:


for train_index,test_index in split.split(X,y):
    train_x=X[train_index]
    train_y=y[train_index]
    test_x=X[test_index]
    test_y=y[test_index]


# # Model training

# In[168]:


from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import cross_val_score,cross_val_predict,GridSearchCV,cross_validate
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score


# In[155]:


def scoring(model,x,y):
    # function that output the cross val score for a model
    print(np.mean(cross_val_score(model,x,y,scoring='f1',cv=3)))


# In[156]:


def matrix(model,x,y):
    # takes model and x,y as input and output confustion matric and classification report
    pred=cross_val_predict(model,x,y,cv=3)
    print(confusion_matrix(y,pred))
    print(classification_report(y,pred))


# ## Logistic Regression

# In[85]:


from sklearn.linear_model import LogisticRegression


# In[86]:


lr1=LogisticRegression()


# In[87]:


matrix(lr1,train_x,train_y)


# In[88]:


scoring(lr1,train_x,train_y)


# ### Training SVC 

# In[157]:


sv=SVC(kernel='linear')
matrix(sv,train_x,train_y)


# In[158]:


scoring(sv,train_x,train_y)


# In[159]:


sv2=SVC(kernel='rbf',)


# In[160]:


matrix(sv2,train_x,train_y)


# In[161]:


scoring(sv2,train_x,train_y)


# In[162]:


# Param grid with different c and gamma values with a gaussian kernel
param_grid={
    'C':[0.001,0.01,0.1,1,5,10,15],
    'gamma':[0.001,0.01,0.1,1,3,5,9],
    'kernel':['rbf']
}


# In[163]:


sv=SVC()
sv_grid_g=GridSearchCV(sv,param_grid=param_grid,scoring='f1')


# In[164]:


sv_grid_g.fit(train_x,train_y)


# In[165]:


result_gamma=sv_grid_g.cv_results_


# In[166]:


for i, j in zip(result_gamma['mean_test_score'],result_gamma['params']):
    print(i,j)


# In[167]:


# The precision and recall values have improved with the gridsearch
print(confusion_matrix(test_y,sv_grid_g.predict(test_x)),classification_report(test_y,sv_grid_g.predict(test_x)))


# ### Accuracy score for svm with rbf kernel

# In[73]:


# Accuracy score test set
accuracy_score(test_y,sv_grid_g.predict(test_x))


# In[74]:


# Accuracy score with training set
accuracy_score(train_y,sv_grid_g.predict(train_x))


# In[170]:


#F1score - testing data
f1_score(test_y,sv_grid_g.predict(test_x))


# In[171]:


#F1score - training data
f1_score(train_y,sv_grid_g.predict(train_x))


# In[172]:


# parameter for grid search on SVC with no kernel
param_grid_linear={
    'C':[0.001,0.01,0.1,1,5,10,15],
    'gamma':[0.001,0.01,0.1,1,3,5],
    'kernel':['linear']
}


# In[173]:


sv=SVC()
sv_grid_linear=GridSearchCV(sv,param_grid=param_grid_linear,scoring='f1')


# In[174]:


sv_grid_linear.fit(train_x,train_y)


# In[175]:


result_linear=sv_grid_linear.cv_results_
for i, j in zip(result_linear['mean_test_score'],result_linear['params']):
    print(i,j)


# ### Accuracy score for svm with no kernel

# In[176]:


# accuracy score with training set
accuracy_score(train_y,sv_grid_linear.predict(train_x))


# In[177]:


# accuracy score with test set
accuracy_score(test_y,sv_grid_linear.predict(test_x))


# In[179]:


#F1score - testing data
f1_score(test_y,sv_grid_linear.predict(test_x))


# In[180]:


#F1score - training data
f1_score(train_y,sv_grid_linear.predict(train_x))


# In[178]:


# Shows the weight for each coefficients
for i in zip(columns_post,sv_grid_linear.best_estimator_.coef_.flatten()):
    print(i)


# In[82]:


def to_csv(filename,model):
    # a function that predicts the test set provided and cave it in csv file 
    tes=pd.read_csv('test_lAUu6dG.csv')
    x,column=preprocessing(tes,target=False)
    new=tes[['Loan_ID']]
    new['Loan_Status']=model.predict(x)
    new.Loan_Status=new.Loan_Status.apply(lambda x : 'Y' if x==1 else 'N')
    new.to_csv(filename,index=False)


# In[83]:


to_csv('sv1.csv',sv_grid_g)


# In[84]:


to_csv('sv_linear.csv',sv_grid_linear)


# ## Descision tree

# In[181]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[182]:


dt=DecisionTreeClassifier()


# In[183]:


matrix(dt,train_x,train_y)


# In[184]:


train_x.shape


# In[185]:


scoring(dt,train_x,train_y)


# In[186]:


dt.fit(train_x,train_y)


# In[187]:


imp=pd.DataFrame(columns_post,columns=['name'])


# In[ ]:





# In[188]:


imp['dtree']=dt.feature_importances_


# In[189]:


imp.set_index('name',inplace=True)


# In[190]:


imp.dtree.sort_values(ascending=False).plot(kind='bar')


# In[191]:


# param grid to train a decision tree with different depth 
param_grid={
    'max_depth':[2,6,10,12,18,26,30,36]
}


# In[193]:


dt_grid=GridSearchCV(dt,param_grid=param_grid,scoring='f1',cv=3)


# In[194]:


dt_grid.fit(train_x,train_y)


# In[195]:


result=dt_grid.cv_results_


# In[196]:


for i, j in zip(result['mean_test_score'],result['params']):
    print(i,j)


# In[197]:


imp['dtree_max_depth']=dt_grid.best_estimator_.feature_importances_


# In[198]:


imp.dtree_max_depth.sort_values(ascending=False).plot(kind='bar')
# As only 5-6 features have comparetivley higher feature importance lets try to use PCA to reduce the dimension


# In[200]:


from sklearn.decomposition import PCA 
pca=PCA(n_components=5)


# In[201]:


dt=DecisionTreeClassifier()
decision_pipe=Pipeline([
    ('pca',pca),
    ('dtree',dt)
])


# In[205]:


param_grid_desc2={
    'pca__n_components':[2,4,6,10,12,14],
    'dtree__max_depth':[2,4,6,8,10,12,18,26,30],
    'dtree__min_samples_split':[2,4,6,10],
    'dtree__criterion':['gini','entropy'],
    'dtree__min_samples_leaf':[1,2,3,4,5]
}


# In[206]:


grid_dim_desc=GridSearchCV(decision_pipe,param_grid=param_grid_desc2,cv=3,scoring='f1')


# In[207]:


grid_dim_desc.fit(train_x,train_y)


# In[208]:


# the best estimator consists of 12 components after pca
grid_dim_desc.best_estimator_


# In[209]:


result_dimen=grid_dim_desc.cv_results_
for i, j in zip(result_dimen['mean_test_score'],result_dimen['params']):
    print(i,j)


# In[210]:


print(confusion_matrix(train_y,grid_dim_desc.predict(train_x)),classification_report(train_y,grid_dim_desc.predict(train_x)),accuracy_score(train_y,grid_dim_desc.predict(train_x)))


# In[211]:


print(confusion_matrix(test_y,grid_dim_desc.predict(test_x)),classification_report(test_y,grid_dim_desc.predict(test_x)),accuracy_score(test_y,grid_dim_desc.predict(test_x)))


# In[212]:


import joblib
joblib.dump(grid_dim_desc,'decision_tree_trained')
import joblib
model=joblib.load('decision_tree_trained')


# In[ ]:





# ## accuracy score for training set

# In[213]:


print(accuracy_score(train_y,model.predict(train_x)))


# ## accuracy score for test set

# In[214]:


print(accuracy_score(test_y,model.predict(test_x)))


# In[215]:


to_csv('decision_tree.csv',grid_dim_desc)


# # Random forest

# In[216]:


from sklearn.ensemble import RandomForestClassifier


# In[217]:


rfc=RandomForestClassifier()


# In[218]:


matrix(rfc,train_x,train_y)


# In[219]:


scoring(rfc,train_x,train_y)


# In[220]:


rfc.fit(train_x,train_y)


# In[221]:


imp['rfc']=rfc.feature_importances_


# In[222]:


imp['rfc'].sort_values(ascending=False).plot(kind='bar')


# In[223]:


rfc=RandomForestClassifier()


# In[224]:


param_grid_rfc={
    'max_depth':[2,4,6],
    'n_estimators':[500],
    'criterion':['gini','entropy'],
    
}


# In[225]:


grid_rfc=GridSearchCV(rfc,param_grid=param_grid_rfc,scoring='f1',cv=3)


# In[226]:


grid_rfc.fit(train_x,train_y)


# In[227]:


grid_rfc.best_estimator_


# In[228]:


result_rfc=grid_rfc.cv_results_
for i, j in zip(result_rfc['mean_test_score'],result_rfc['params']):
    print(i,j)


# In[229]:


param_grid_rfc_2={
    
    'n_estimators':[500,300,200],

}


# In[230]:


rfc=RandomForestClassifier(max_depth=2)
grid_rfc2=GridSearchCV(rfc,param_grid=param_grid_rfc_2,scoring='f1',cv=3)
grid_rfc2.fit(train_x,train_y)


# In[231]:


result_rfc=grid_rfc2.cv_results_
for i, j in zip(result_rfc['mean_test_score'],result_rfc['params']):
    print(i,j)


# ## accuracy score for training set

# In[232]:


print(accuracy_score(train_y,grid_rfc2.predict(train_x)))


# ## accuracy score for training set

# In[233]:


print(accuracy_score(test_y,grid_rfc2.predict(test_x)))


# In[234]:


to_csv('random_forest.csv',grid_rfc)


# # Ensemble voting classifier (svm,logistic regression, decision tree)

# In[235]:


from sklearn.ensemble import VotingClassifier


# In[236]:


log=LogisticRegression()
svm=SVC(gamma=0.01)
des=DecisionTreeClassifier(criterion='entropy',max_depth=2)
vot_clf=VotingClassifier(estimators=[('log',log),('svm',svm),('desc',des)])


# In[237]:


vot_clf.fit(train_x,train_y)


# In[238]:


print(accuracy_score(test_y,vot_clf.predict(test_x)))


# In[239]:


# bagging with logisticregression, decison tree, svc with PCA
log=LogisticRegression()
svm=SVC(gamma=0.01)
des=DecisionTreeClassifier(criterion='entropy',max_depth=2)
pca=PCA(n_components=12)
des_pipe=Pipeline([
    ('pca',pca),
    ('dtree',des)
])

vot_clf_2=VotingClassifier(estimators=[('log',log),('svm',svm),('desc',des_pipe)])


# In[240]:


vot_clf_2.fit(train_x,train_y)


# In[241]:


print(accuracy_score(test_y,vot_clf_2.predict(test_x)))


# In[242]:


log=LogisticRegression()
svm=SVC(gamma=0.01,probability=True)
des=DecisionTreeClassifier(criterion='entropy',max_depth=2)
rft=RandomForestClassifier()
vot_clf_3=VotingClassifier(estimators=[('log',log),('svm',svm),('desc',des),('rft',rft)],voting='soft')


# In[243]:


vot_clf_3.fit(train_x,train_y)


# In[244]:


print(accuracy_score(test_y,vot_clf_3.predict(test_x)))


# In[245]:


print(accuracy_score(train_y,vot_clf_3.predict(train_x)),'\n',classification_report(train_y,vot_clf_3.predict(train_x)))


# In[246]:


to_csv('voting.csv',vot_clf_3)


# # Ensemble Bagging

# In[247]:


from sklearn.ensemble import BaggingClassifier


# ### Bagging with SVM

# In[254]:


sv1=SVC()
bag4=BaggingClassifier(SVC(),n_estimators=500,max_samples=100,n_jobs=-1)


# In[255]:


bag4.fit(train_x,train_y)


# In[256]:


accuracy_score(train_y,bag4.predict(train_x))


# In[257]:


accuracy_score(test_y,bag4.predict(test_x))


# In[ ]:




