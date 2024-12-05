# %% [markdown]
# ## House Price Prediction 
# - - We all have experienced a time when we have to look up for a new house to buy. But then the journey begins with a lot of frauds, negotiating deals, researching the local areas and so on.

# %% [markdown]
# #### So to deal with this kind of issues Today we will be preparing a MACHINE LEARNING Based model, trained on the House Price Prediction Dataset.

# %%
# install the packages needed



# %%
# import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
# load the dataset and check the info
df = pd.read_excel("./HousePricePrediction.xlsx")
df.info()

# %%
# display first 5
df.head()

# %% [markdown]
# 1	Id	To count the records.
# 
# 2	MSSubClass	 Identifies the type of dwelling involved in the sale.
# 
# 3	MSZoning	Identifies the general zoning classification of the sale.
# 
# 4	LotArea	 Lot size in square feet.
# 
# 5	LotConfig	Configuration of the lot
# 
# 6	BldgType	Type of dwelling
# 
# 7	OverallCond	Rates the overall condition of the house
# 
# 8	YearBuilt	Original construction year
# 
# 9	YearRemodAdd	Remodel date (same as construction date if no remodeling or additions).
# 
# 10	Exterior1st	Exterior covering on house
# 
# 11	BsmtFinSF2	Type 2 finished square feet.
# 
# 12	TotalBsmtSF	Total square feet of basement area
# 
# 13	SalePrice	To be predicted

# %%
# describe
df.describe()

# %%
# check the numbers of unique values in each category
df.nunique()

# %%
for col in df.columns:
    print(col)
    print(df[col].unique())

# %%
df.isnull().sum()

# %%
# number of entries
df.shape[0]

# %% [markdown]
# ### Clean data

# %%
# lower and replace space with underscore
df.columns = df.columns.str.lower().str.replace(" ", "_")
df.columns

# %%
# seprate numerical values from categorical values
df_numerical = df.select_dtypes(include=np.number).columns
df_category = df.select_dtypes(include=object).columns

# %%
df_numerical, df_category


# %%
# lower and replace space with underscore for each categorical colunm values
for col in df_category:
    df[col] = df[col].str.lower().str.replace(" ", "_")

# %%
# check the changes
df.head()

# %%
# check null values

df.isnull().sum()

# %%
# check the datatypes present 

obj = (df.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (df.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (df.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))

# %% [markdown]
# ### EDA

# %%
import seaborn as sns


# %%
# seprate numerical values from categorical values
df_numerical = df.select_dtypes(include=np.number).columns
df_category = df.select_dtypes(include=object).columns

df_numerical, df_category


# %%
# heatmap to visualize relationship
plt.figure(figsize=(12,6))
sns.heatmap(df[df_numerical].corr(), cmap="BrBG", fmt=".2f", linewidths="2", annot=True)

# %%
unique_values = []

for col in df_category:
    unique_values.append(df[col].unique().size)
    
    plt.figure(figsize=(10,6))
    plt.title("No. of  unique values of categorical value")
    plt.xticks(rotation= 90)
    sns.barplot(x = df_category, y = unique_values)

# %%
# unique_values = [] for col in df_category: if col in df.columns: # Check if column exists in dataframe unique_values.append(df[col].nunique()) if len(df_category) == len(unique_values): # Ensure lengths match plt.figure(figsize=(10, 6)) plt.title("No. of Unique Values of Categorical Columns") plt.xticks(rotation=90) sns.barplot(x=df_category, y=unique_values) plt.xlabel('Categorical Columns') plt.ylabel('Number of Unique Values') plt.show() else: print("The lengths of df_category and unique_values do not match.")

# %%

unique_values = []

for col in df_category:
    if col in df.columns:  # Check if column exists in dataframe
        unique_values.append(df[col].nunique())

if len(df_category) == len(unique_values):  # Ensure lengths match
    plt.figure(figsize=(10, 6))
    plt.title("No. of Unique Values of Categorical Columns")
    plt.xticks(rotation=90)
    sns.barplot(x=df_category, y=unique_values)
    plt.xlabel('Categorical Columns')
    plt.ylabel('Number of Unique Values')
    plt.show()
else:
    print("The lengths of df_category and unique_values do not match.")


# %%
plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in df[df_category]:
    y = df[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1

# %%
df = df.drop(['Id'], axis=1)

# %%
df.head()

# %%
df['SalePrice'].isnull().sum()

# %%
df["SalePrice"] = df['SalePrice'].fillna(df['SalePrice'].mean())

# %%
df['SalePrice'].iloc[::20]

# %%
df['SalePrice'].isnull().sum()

# %%
new_df = df.dropna()

# %%
new_df.isnull().sum()

# %%
new_df

# %%
new_df.columns = new_df.columns.str.lower().str.replace(" ", "_")

# %%
new_df.dtypes

# %%
for col in new_df[df_category].columns:
    new_df = new_df[col].str.lower().str.replace(" ", "_")

# %%
from sklearn.model_selection import train_test_split
# split the dataset
df_train_full, df_test = train_test_split(new_df ,test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

# %%
len(df_train), len(df_val), len(df_test)

# %%
df_train

# %%
# extract the target vairable

y_train = df_train["saleprice"]
y_test = df_test["saleprice"]
y_val = df_val["saleprice"]



# %%
# del col
del df_train["saleprice"]
del df_test["saleprice"]
del df_val["saleprice"]




# %%
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient = "records")
val_dict = df_val.to_dict(orient = "records")
X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)
len(X_train), len(X_val)


# %%
dv.get_feature_names_out()

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error

# using random forest regressor


model = RandomForestRegressor(n_estimators = 10)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mean_b_value = round(mean_absolute_percentage_error(y_val, y_pred),2)
print("RandomForestRegressor model:", mean_b_value)

# %%
n_values = [10, 20, 100, 30, 50]

for n in n_values:
    model_r = RandomForestRegressor(random_state=1, n_jobs=-1, n_estimators=n)
    model_r.fit(X_train, y_train)
    y_pred_r = model.predict(X_val)
    print(n)
    print("mean_absolute_percentage_error", round(mean_absolute_percentage_error(y_val, y_pred_r), 2))
    print("root_mean_squared_error", round(root_mean_squared_error(y_val, y_pred_r),2))

# %%
from sklearn.linear_model import LinearRegression

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_val)
mean_b_value_ln = round(mean_absolute_percentage_error(y_val, y_pred_linear),2)
print("LinearRegression model ", mean_b_value_ln)

# %% [markdown]
# ### Hyperparameter 

# %%
from sklearn.metrics import r2_score

print(round(r2_score(y_val, y_pred_r), 2))

# %%
print(round(r2_score(y_val, y_pred_linear), 2))

# %%
print(r2_score(y_val, y_pred))

# %%
# save model

import pickle as pk
with open("model_linear.bin", "wb") as f_out:
    pk.dump((dv, model_linear), f_out)

# %%
# load model
import pickle as pk

with open("model_linear.bin", "rb") as f_in:
    dv, model = pk.load(f_in)

# %%


# %%
import streamlit


# %%
dv

# %%
model

# %%


# %%


