import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(color_codes=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.416219Z","iopub.execute_input":"2024-03-02T02:25:47.416642Z","iopub.status.idle":"2024-03-02T02:25:47.443278Z","shell.execute_reply.started":"2024-03-02T02:25:47.416604Z","shell.execute_reply":"2024-03-02T02:25:47.442053Z"}}
train = pd.read_csv("/kaggle/input/finance-company-loan-data/train_ctrUa4K.csv")
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.445299Z","iopub.execute_input":"2024-03-02T02:25:47.445618Z","iopub.status.idle":"2024-03-02T02:25:47.466909Z","shell.execute_reply.started":"2024-03-02T02:25:47.445589Z","shell.execute_reply":"2024-03-02T02:25:47.465793Z"}}
test = pd.read_csv("/kaggle/input/finance-company-loan-data/test_lAUu6dG.csv")
test.head()

# %% [markdown]
# # **Data Processing Part1**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.468518Z","iopub.execute_input":"2024-03-02T02:25:47.468868Z","iopub.status.idle":"2024-03-02T02:25:47.495705Z","shell.execute_reply.started":"2024-03-02T02:25:47.468837Z","shell.execute_reply":"2024-03-02T02:25:47.494509Z"}}
train.drop(columns=["Loan_ID"], inplace=True)
train.head()

# %% [markdown]
# # finding the percentage of missing values columnwise

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.498533Z","iopub.execute_input":"2024-03-02T02:25:47.498929Z","iopub.status.idle":"2024-03-02T02:25:47.511821Z","shell.execute_reply.started":"2024-03-02T02:25:47.498895Z","shell.execute_reply":"2024-03-02T02:25:47.510922Z"}}
missing = train.isnull().sum()*100/ train.shape[0]
missing[missing>0].sort_values(ascending=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.513172Z","iopub.execute_input":"2024-03-02T02:25:47.513467Z","iopub.status.idle":"2024-03-02T02:25:47.520864Z","shell.execute_reply.started":"2024-03-02T02:25:47.513441Z","shell.execute_reply":"2024-03-02T02:25:47.519564Z"}}
train.columns

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.522262Z","iopub.execute_input":"2024-03-02T02:25:47.522594Z","iopub.status.idle":"2024-03-02T02:25:47.535081Z","shell.execute_reply.started":"2024-03-02T02:25:47.522557Z","shell.execute_reply":"2024-03-02T02:25:47.533620Z"}}
train['Credit_History'].fillna(0,inplace=True)
train['Self_Employed'].fillna('No',inplace=True)
train['LoanAmount'].fillna(0,inplace=True)
train['Dependents'].fillna('Other',inplace=True)
train['Loan_Amount_Term'].fillna(0,inplace=True)
train['Gender'].fillna('Other',inplace=True)
train['Married'].fillna('Other',inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.536756Z","iopub.execute_input":"2024-03-02T02:25:47.537207Z","iopub.status.idle":"2024-03-02T02:25:47.553129Z","shell.execute_reply.started":"2024-03-02T02:25:47.537172Z","shell.execute_reply":"2024-03-02T02:25:47.551892Z"}}
missing = train.isnull().sum()*100/ train.shape[0]
missing[missing>0].sort_values(ascending=False)

# %% [markdown]
# # Exploratory Data Analysis

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.554727Z","iopub.execute_input":"2024-03-02T02:25:47.555466Z","iopub.status.idle":"2024-03-02T02:25:47.846233Z","shell.execute_reply.started":"2024-03-02T02:25:47.555421Z","shell.execute_reply":"2024-03-02T02:25:47.845132Z"}}
sns.countplot(data=train,x='Loan_Status',hue ='Property_Area')

# %% [markdown]
# people living in suburban has high acceptable chance of loan status

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:47.847668Z","iopub.execute_input":"2024-03-02T02:25:47.848067Z","iopub.status.idle":"2024-03-02T02:25:48.073993Z","shell.execute_reply.started":"2024-03-02T02:25:47.848034Z","shell.execute_reply":"2024-03-02T02:25:48.072874Z"}}
sns.countplot(data=train,x='Loan_Status',hue ='Credit_History')

# %% [markdown]
# from this we can infer that, people with acceptable past credit history are most likely accepted in new loan.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:48.079422Z","iopub.execute_input":"2024-03-02T02:25:48.079836Z","iopub.status.idle":"2024-03-02T02:25:48.448578Z","shell.execute_reply.started":"2024-03-02T02:25:48.079797Z","shell.execute_reply":"2024-03-02T02:25:48.447246Z"}}
sns.countplot(data=train,x='Loan_Status',hue ='Loan_Amount_Term')

# %% [markdown]
# people with 360 month loan term are mostly to be accepted

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:48.449987Z","iopub.execute_input":"2024-03-02T02:25:48.450439Z","iopub.status.idle":"2024-03-02T02:25:48.716295Z","shell.execute_reply.started":"2024-03-02T02:25:48.450406Z","shell.execute_reply":"2024-03-02T02:25:48.715467Z"}}
sns.barplot(data=train,x='Loan_Status',y ='LoanAmount')

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:48.717565Z","iopub.execute_input":"2024-03-02T02:25:48.718058Z","iopub.status.idle":"2024-03-02T02:25:48.995239Z","shell.execute_reply.started":"2024-03-02T02:25:48.718025Z","shell.execute_reply":"2024-03-02T02:25:48.994046Z"}}
sns.barplot(data=train,x='Loan_Status',y ='CoapplicantIncome')

# %% [markdown]
# people with high coapplicant income are mostly accepted to new loan

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:48.996512Z","iopub.execute_input":"2024-03-02T02:25:48.996893Z","iopub.status.idle":"2024-03-02T02:25:49.307822Z","shell.execute_reply.started":"2024-03-02T02:25:48.996861Z","shell.execute_reply":"2024-03-02T02:25:49.306489Z"}}
sns.barplot(data=train,x='Loan_Status',y='ApplicantIncome')

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:49.309719Z","iopub.execute_input":"2024-03-02T02:25:49.310878Z","iopub.status.idle":"2024-03-02T02:25:49.560803Z","shell.execute_reply.started":"2024-03-02T02:25:49.310828Z","shell.execute_reply":"2024-03-02T02:25:49.559495Z"}}
sns.countplot(data=train,x='Loan_Status',hue ='Self_Employed')

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:49.562288Z","iopub.execute_input":"2024-03-02T02:25:49.563019Z","iopub.status.idle":"2024-03-02T02:25:49.809980Z","shell.execute_reply.started":"2024-03-02T02:25:49.562975Z","shell.execute_reply":"2024-03-02T02:25:49.808786Z"}}
sns.countplot(data=train,x='Loan_Status',hue ='Education')

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:49.811689Z","iopub.execute_input":"2024-03-02T02:25:49.812559Z","iopub.status.idle":"2024-03-02T02:25:50.089983Z","shell.execute_reply.started":"2024-03-02T02:25:49.812510Z","shell.execute_reply":"2024-03-02T02:25:50.088662Z"}}
sns.countplot(data=train,x='Loan_Status',hue ='Dependents')

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.091676Z","iopub.execute_input":"2024-03-02T02:25:50.093061Z","iopub.status.idle":"2024-03-02T02:25:50.339834Z","shell.execute_reply.started":"2024-03-02T02:25:50.093014Z","shell.execute_reply":"2024-03-02T02:25:50.338716Z"}}
sns.countplot(data=train,x='Loan_Status',hue ='Married')

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.341236Z","iopub.execute_input":"2024-03-02T02:25:50.341590Z","iopub.status.idle":"2024-03-02T02:25:50.597969Z","shell.execute_reply.started":"2024-03-02T02:25:50.341559Z","shell.execute_reply":"2024-03-02T02:25:50.597064Z"}}
sns.countplot(data=train,x='Loan_Status',hue ='Gender')

# %% [markdown]
# 
# 
# # Data Preprocessing Part-II

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.599175Z","iopub.execute_input":"2024-03-02T02:25:50.600130Z","iopub.status.idle":"2024-03-02T02:25:50.666645Z","shell.execute_reply.started":"2024-03-02T02:25:50.600096Z","shell.execute_reply":"2024-03-02T02:25:50.665125Z"}}
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.668219Z","iopub.execute_input":"2024-03-02T02:25:50.668571Z","iopub.status.idle":"2024-03-02T02:25:50.695148Z","shell.execute_reply.started":"2024-03-02T02:25:50.668537Z","shell.execute_reply":"2024-03-02T02:25:50.694275Z"}}
train

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.696638Z","iopub.execute_input":"2024-03-02T02:25:50.697269Z","iopub.status.idle":"2024-03-02T02:25:50.707580Z","shell.execute_reply.started":"2024-03-02T02:25:50.697232Z","shell.execute_reply":"2024-03-02T02:25:50.706610Z"}}
train['Gender'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.709104Z","iopub.execute_input":"2024-03-02T02:25:50.710237Z","iopub.status.idle":"2024-03-02T02:25:50.719540Z","shell.execute_reply.started":"2024-03-02T02:25:50.710202Z","shell.execute_reply":"2024-03-02T02:25:50.718526Z"}}
train['Married'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.720912Z","iopub.execute_input":"2024-03-02T02:25:50.721245Z","iopub.status.idle":"2024-03-02T02:25:50.731021Z","shell.execute_reply.started":"2024-03-02T02:25:50.721215Z","shell.execute_reply":"2024-03-02T02:25:50.729594Z"}}
train['Dependents'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.732474Z","iopub.execute_input":"2024-03-02T02:25:50.733011Z","iopub.status.idle":"2024-03-02T02:25:50.742014Z","shell.execute_reply.started":"2024-03-02T02:25:50.732977Z","shell.execute_reply":"2024-03-02T02:25:50.740983Z"}}
train['Education'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.743193Z","iopub.execute_input":"2024-03-02T02:25:50.743682Z","iopub.status.idle":"2024-03-02T02:25:50.752968Z","shell.execute_reply.started":"2024-03-02T02:25:50.743651Z","shell.execute_reply":"2024-03-02T02:25:50.752185Z"}}
train['Self_Employed'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.754142Z","iopub.execute_input":"2024-03-02T02:25:50.754611Z","iopub.status.idle":"2024-03-02T02:25:50.765316Z","shell.execute_reply.started":"2024-03-02T02:25:50.754580Z","shell.execute_reply":"2024-03-02T02:25:50.764520Z"}}
train['Property_Area'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.766537Z","iopub.execute_input":"2024-03-02T02:25:50.767361Z","iopub.status.idle":"2024-03-02T02:25:50.776939Z","shell.execute_reply.started":"2024-03-02T02:25:50.767320Z","shell.execute_reply":"2024-03-02T02:25:50.775819Z"}}
train['Loan_Status'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.785138Z","iopub.execute_input":"2024-03-02T02:25:50.785919Z","iopub.status.idle":"2024-03-02T02:25:50.793517Z","shell.execute_reply.started":"2024-03-02T02:25:50.785882Z","shell.execute_reply":"2024-03-02T02:25:50.792429Z"}}
train['Loan_Amount_Term'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.794723Z","iopub.execute_input":"2024-03-02T02:25:50.795379Z","iopub.status.idle":"2024-03-02T02:25:50.805336Z","shell.execute_reply.started":"2024-03-02T02:25:50.795345Z","shell.execute_reply":"2024-03-02T02:25:50.804209Z"}}
train['Gender'] = label_encoder.fit_transform(train['Gender'])
train['Gender'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.806945Z","iopub.execute_input":"2024-03-02T02:25:50.807617Z","iopub.status.idle":"2024-03-02T02:25:50.817886Z","shell.execute_reply.started":"2024-03-02T02:25:50.807574Z","shell.execute_reply":"2024-03-02T02:25:50.817006Z"}}
train['Married'] = label_encoder.fit_transform(train['Married'])
train['Married'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.819227Z","iopub.execute_input":"2024-03-02T02:25:50.819704Z","iopub.status.idle":"2024-03-02T02:25:50.828934Z","shell.execute_reply.started":"2024-03-02T02:25:50.819663Z","shell.execute_reply":"2024-03-02T02:25:50.828068Z"}}
train['Dependents']=label_encoder.fit_transform(train['Dependents'])
train['Dependents'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.830271Z","iopub.execute_input":"2024-03-02T02:25:50.830834Z","iopub.status.idle":"2024-03-02T02:25:50.840588Z","shell.execute_reply.started":"2024-03-02T02:25:50.830782Z","shell.execute_reply":"2024-03-02T02:25:50.839814Z"}}
train['Education'] = label_encoder.fit_transform(train['Education'])
train["Education"].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.841894Z","iopub.execute_input":"2024-03-02T02:25:50.842648Z","iopub.status.idle":"2024-03-02T02:25:50.854714Z","shell.execute_reply.started":"2024-03-02T02:25:50.842605Z","shell.execute_reply":"2024-03-02T02:25:50.853914Z"}}
train['Self_Employed'] = label_encoder.fit_transform(train['Self_Employed'])
train['Self_Employed'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.856113Z","iopub.execute_input":"2024-03-02T02:25:50.857022Z","iopub.status.idle":"2024-03-02T02:25:50.867223Z","shell.execute_reply.started":"2024-03-02T02:25:50.856988Z","shell.execute_reply":"2024-03-02T02:25:50.866489Z"}}
train['Property_Area'] = label_encoder.fit_transform(train['Property_Area'])
train['Property_Area'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.868367Z","iopub.execute_input":"2024-03-02T02:25:50.869024Z","iopub.status.idle":"2024-03-02T02:25:50.878609Z","shell.execute_reply.started":"2024-03-02T02:25:50.868978Z","shell.execute_reply":"2024-03-02T02:25:50.877498Z"}}
train['Loan_Amount_Term'] = label_encoder.fit_transform(train['Loan_Amount_Term'])
train['Loan_Amount_Term'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.880099Z","iopub.execute_input":"2024-03-02T02:25:50.880823Z","iopub.status.idle":"2024-03-02T02:25:50.889612Z","shell.execute_reply.started":"2024-03-02T02:25:50.880780Z","shell.execute_reply":"2024-03-02T02:25:50.888893Z"}}
train['Loan_Status'] = label_encoder.fit_transform(train['Loan_Status'])
train['Loan_Status'].unique()

# %% [markdown]
# # **Check the  Outlier**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:50.890974Z","iopub.execute_input":"2024-03-02T02:25:50.891499Z","iopub.status.idle":"2024-03-02T02:25:51.147999Z","shell.execute_reply.started":"2024-03-02T02:25:50.891465Z","shell.execute_reply":"2024-03-02T02:25:51.146810Z"}}
sns.boxplot(x=train['ApplicantIncome'])

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:51.149220Z","iopub.execute_input":"2024-03-02T02:25:51.149527Z","iopub.status.idle":"2024-03-02T02:25:51.331231Z","shell.execute_reply.started":"2024-03-02T02:25:51.149499Z","shell.execute_reply":"2024-03-02T02:25:51.330102Z"}}
sns.boxplot(x=train['CoapplicantIncome'])

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:51.332882Z","iopub.execute_input":"2024-03-02T02:25:51.333571Z","iopub.status.idle":"2024-03-02T02:25:51.537046Z","shell.execute_reply.started":"2024-03-02T02:25:51.333523Z","shell.execute_reply":"2024-03-02T02:25:51.535950Z"}}
sns.boxplot(x=train['LoanAmount'])

# %% [code]


# %% [markdown]
# # Balanced Class Data

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:51.538511Z","iopub.execute_input":"2024-03-02T02:25:51.538991Z","iopub.status.idle":"2024-03-02T02:25:51.558023Z","shell.execute_reply.started":"2024-03-02T02:25:51.538946Z","shell.execute_reply":"2024-03-02T02:25:51.556922Z"}}
import scipy.stats as stats
z = np.abs(stats.zscore(train))
data_clean = train[(z<3).all(axis = 1)]
data_clean.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:51.559710Z","iopub.execute_input":"2024-03-02T02:25:51.561057Z","iopub.status.idle":"2024-03-02T02:25:51.731257Z","shell.execute_reply.started":"2024-03-02T02:25:51.561010Z","shell.execute_reply":"2024-03-02T02:25:51.730071Z"}}
sns.countplot(data = data_clean, x='Loan_Status')
data_clean['Loan_Status'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:51.733262Z","iopub.execute_input":"2024-03-02T02:25:51.733757Z","iopub.status.idle":"2024-03-02T02:25:51.746466Z","shell.execute_reply.started":"2024-03-02T02:25:51.733686Z","shell.execute_reply":"2024-03-02T02:25:51.745180Z"}}
from sklearn.utils import resample
#we are about to create two different dataframe of majority and minority class
df_majority = data_clean[(data_clean['Loan_Status']==1)]
df_minority = data_clean[(data_clean['Loan_Status']==0)]
#now we are upsampling minority class
df_minority_upsampled = resample(df_minority,
                                 replace = True, #sample with replacements
                                 n_samples = 398, #to match majority class
                                 random_state = 0) #reproducible results
#combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled,df_majority])

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:51.748533Z","iopub.execute_input":"2024-03-02T02:25:51.749001Z","iopub.status.idle":"2024-03-02T02:25:51.921835Z","shell.execute_reply.started":"2024-03-02T02:25:51.748955Z","shell.execute_reply":"2024-03-02T02:25:51.920593Z"}}
sns.countplot(data=df_upsampled,x='Loan_Status')
df_upsampled['Loan_Status'].value_counts()

# %% [markdown]
# # Data correlaton

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:51.923204Z","iopub.execute_input":"2024-03-02T02:25:51.924130Z","iopub.status.idle":"2024-03-02T02:25:52.464815Z","shell.execute_reply.started":"2024-03-02T02:25:51.924085Z","shell.execute_reply":"2024-03-02T02:25:52.463665Z"}}
sns.heatmap(df_upsampled.corr(),fmt='2g')

# %% [markdown]
# # **Building Machine Learning Model**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:52.466397Z","iopub.execute_input":"2024-03-02T02:25:52.467026Z","iopub.status.idle":"2024-03-02T02:25:52.475793Z","shell.execute_reply.started":"2024-03-02T02:25:52.466981Z","shell.execute_reply":"2024-03-02T02:25:52.474635Z"}}
x = df_upsampled.drop('Loan_Status', axis=1)
y = df_upsampled['Loan_Status']

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:52.477438Z","iopub.execute_input":"2024-03-02T02:25:52.478441Z","iopub.status.idle":"2024-03-02T02:25:52.539565Z","shell.execute_reply.started":"2024-03-02T02:25:52.478403Z","shell.execute_reply":"2024-03-02T02:25:52.538423Z"}}
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)

# %% [markdown]
# # **Decision Tree**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:52.541053Z","iopub.execute_input":"2024-03-02T02:25:52.541467Z","iopub.status.idle":"2024-03-02T02:25:52.689119Z","shell.execute_reply.started":"2024-03-02T02:25:52.541436Z","shell.execute_reply":"2024-03-02T02:25:52.687633Z"}}
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(x_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:52.692632Z","iopub.execute_input":"2024-03-02T02:25:52.693652Z","iopub.status.idle":"2024-03-02T02:25:52.702945Z","shell.execute_reply.started":"2024-03-02T02:25:52.693607Z","shell.execute_reply":"2024-03-02T02:25:52.701793Z"}}
y_pred = dtree.predict(x_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:52.704358Z","iopub.execute_input":"2024-03-02T02:25:52.705357Z","iopub.status.idle":"2024-03-02T02:25:52.718114Z","shell.execute_reply.started":"2024-03-02T02:25:52.705312Z","shell.execute_reply":"2024-03-02T02:25:52.716790Z"}}
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
print('f-1 score:',(f1_score(y_test,y_pred)))
print('Precision score:',(precision_score(y_test,y_pred)))
print('Recall Score:',(recall_score(y_test,y_pred)))

# %% [markdown]
# # **Random Forest**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:52.719455Z","iopub.execute_input":"2024-03-02T02:25:52.719808Z","iopub.status.idle":"2024-03-02T02:25:52.970453Z","shell.execute_reply.started":"2024-03-02T02:25:52.719776Z","shell.execute_reply":"2024-03-02T02:25:52.969296Z"}}
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rfc.fit(x_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:52.971845Z","iopub.execute_input":"2024-03-02T02:25:52.972243Z","iopub.status.idle":"2024-03-02T02:25:52.997853Z","shell.execute_reply.started":"2024-03-02T02:25:52.972214Z","shell.execute_reply":"2024-03-02T02:25:52.996696Z"}}
y_pred = rfc.predict(x_test)
print("Accuracy:",round(accuracy_score(y_test,y_pred)*100,2),"%")

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:52.999460Z","iopub.execute_input":"2024-03-02T02:25:52.999836Z","iopub.status.idle":"2024-03-02T02:25:53.011100Z","shell.execute_reply.started":"2024-03-02T02:25:52.999803Z","shell.execute_reply":"2024-03-02T02:25:53.009918Z"}}
print('f-1 score:',(f1_score(y_test,y_pred)))
print('Precision score:',(precision_score(y_test,y_pred)))
print('Recall Score:',(recall_score(y_test,y_pred)))

# %% [markdown]
# # **Logistic Regression**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:53.012427Z","iopub.execute_input":"2024-03-02T02:25:53.013200Z","iopub.status.idle":"2024-03-02T02:25:53.056065Z","shell.execute_reply.started":"2024-03-02T02:25:53.013154Z","shell.execute_reply":"2024-03-02T02:25:53.054969Z"}}
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
lr.fit(x_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:53.057346Z","iopub.execute_input":"2024-03-02T02:25:53.057796Z","iopub.status.idle":"2024-03-02T02:25:53.066426Z","shell.execute_reply.started":"2024-03-02T02:25:53.057756Z","shell.execute_reply":"2024-03-02T02:25:53.065307Z"}}
y_pred = lr.predict(x_test)
print("Accuracy Score:",round(accuracy_score(y_test,y_pred)*100,2),"%")

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T02:25:53.067724Z","iopub.execute_input":"2024-03-02T02:25:53.068084Z","iopub.status.idle":"2024-03-02T02:25:53.080489Z","shell.execute_reply.started":"2024-03-02T02:25:53.068053Z","shell.execute_reply":"2024-03-02T02:25:53.079285Z"}}
print('f-1 score:',(f1_score(y_test,y_pred)))
print('Precision score:',(precision_score(y_test,y_pred)))
print('Recall Score:',(recall_score(y_test,y_pred)))
