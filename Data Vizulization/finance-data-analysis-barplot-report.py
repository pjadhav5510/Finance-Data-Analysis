import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../input/finance-data/Finance_data.csv', sep=',')
print(f'Count attributes of dataset: {len(data.columns)}')
print(f'Count rows of dataset: {len(data)}')
data.head()

# %% [markdown]
# # DATA CLEANING

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:03.283879Z","iopub.execute_input":"2024-03-02T00:51:03.284281Z","iopub.status.idle":"2024-03-02T00:51:03.299810Z","shell.execute_reply.started":"2024-03-02T00:51:03.284242Z","shell.execute_reply":"2024-03-02T00:51:03.298605Z"}}
#Get data types of Dataframe columns 
data.info()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:05.928710Z","iopub.execute_input":"2024-03-02T00:51:05.929184Z","iopub.status.idle":"2024-03-02T00:51:05.938951Z","shell.execute_reply.started":"2024-03-02T00:51:05.929147Z","shell.execute_reply":"2024-03-02T00:51:05.937952Z"}}
#Check missing values in the dataset
data.isna().sum()

# %% [markdown]
# # VISUALIZE <a id="VISUALIZE"></a>

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:09.334683Z","iopub.execute_input":"2024-03-02T00:51:09.335296Z","iopub.status.idle":"2024-03-02T00:51:09.573930Z","shell.execute_reply.started":"2024-03-02T00:51:09.335252Z","shell.execute_reply":"2024-03-02T00:51:09.573286Z"}}
#Count of people who related to stock market and investment by gender
plt.style.use('ggplot')
data['gender'].value_counts().plot(kind='bar', figsize=(8,5), ylabel='Count of People', xlabel='Gender')
plt.legend()
plt.show()

# %% [markdown]
# **From dataset male are more likely to invest than female**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:16.938024Z","iopub.execute_input":"2024-03-02T00:51:16.938819Z","iopub.status.idle":"2024-03-02T00:51:17.229690Z","shell.execute_reply.started":"2024-03-02T00:51:16.938773Z","shell.execute_reply":"2024-03-02T00:51:17.228793Z"}}
#Show frequency age range by gender 
data.groupby('gender').age.plot(kind='kde')
plt.xlabel('age')
plt.legend()
plt.show()

# %% [markdown]
# **The graph show the most people invest between 25-31 years (kernal density estimation maybe a little error)**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:19.777078Z","iopub.execute_input":"2024-03-02T00:51:19.777409Z","iopub.status.idle":"2024-03-02T00:51:20.211270Z","shell.execute_reply.started":"2024-03-02T00:51:19.777365Z","shell.execute_reply":"2024-03-02T00:51:20.210342Z"}}
data.iloc[:, 3:10].sum().plot(kind='bar')
plt.title('The sum of types of investments')
plt.ylabel('preference (less like)')
plt.show()

# %% [markdown]
# **The rank in order of preferrence (1 like most..7 least like) the data show people prefer to invest Public provident fund.**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:24.225400Z","iopub.execute_input":"2024-03-02T00:51:24.226223Z","iopub.status.idle":"2024-03-02T00:51:24.446381Z","shell.execute_reply.started":"2024-03-02T00:51:24.226180Z","shell.execute_reply":"2024-03-02T00:51:24.445458Z"}}
data['Factor'].value_counts().plot(kind='bar')
plt.ylabel('counts')
plt.title('Factor of Investment')
plt.show()

# %% [markdown]
# **Often factor considered investing in any instrument is "Returns"**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:28.009216Z","iopub.execute_input":"2024-03-02T00:51:28.009672Z","iopub.status.idle":"2024-03-02T00:51:28.453902Z","shell.execute_reply.started":"2024-03-02T00:51:28.009630Z","shell.execute_reply":"2024-03-02T00:51:28.452921Z"}}
#Duration of invest and invest monitor
fig, axes = plt.subplots(1, 2, figsize=(15,5))

sns.countplot(ax=axes[0], x=data['Duration'])
axes[0].set_title('Prefer to keep your money')

sns.countplot(ax=axes[1], x=data['Invest_Monitor'])
axes[1].set_title('Invest Monitor')


# %% [markdown]
# **Guide: [subplotting](https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8)**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:32.232557Z","iopub.execute_input":"2024-03-02T00:51:32.232862Z","iopub.status.idle":"2024-03-02T00:51:32.520588Z","shell.execute_reply.started":"2024-03-02T00:51:32.232829Z","shell.execute_reply":"2024-03-02T00:51:32.519584Z"}}
sns.countplot(x=data['Duration'], hue=data['Invest_Monitor'], palette='coolwarm', edgecolor='black')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# **People usally hold there assets any instruments 1-3 years, 3-5 years and often monitor there investment monthly**

# %% [code] {"execution":{"iopub.status.busy":"2024-03-02T00:51:35.076875Z","iopub.execute_input":"2024-03-02T00:51:35.077174Z","iopub.status.idle":"2024-03-02T00:51:35.339422Z","shell.execute_reply.started":"2024-03-02T00:51:35.077142Z","shell.execute_reply":"2024-03-02T00:51:35.338554Z"}}
sns.countplot(x=data['Source'])

# %% [markdown]
# **Rank information for investment**
# 1. Financial Consultants
# 2. Newspapers and Magazines
# 3. Television
# 4. Internet

# %% [markdown]
# # CONCLUSION <a id="CONCLUSION"></a>

# %% [markdown]
# People who are investors in the dataset are between the ages of 25 and 31. It is mostly male. They're also likely to invest in Gold and Debentures.
# The length of time that money is held in any investment instrument determines whether it is middle-term or long-term investing. In addition, most people keep track of their finances on a monthly basis.
