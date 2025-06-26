#!/usr/bin/env python
# coding: utf-8

# # Import software libraries

# In[387]:


import sys                                                  # Read system parameters.
import numpy as np                                          # Work with multi-dimensional arrays.
import pandas as pd                                         # Manipulate and analyze data.
import scipy as sp                                          # Apply advanced mathematical functions.
from scipy import stats
import matplotlib                                           # Create and format charts.
import matplotlib.pyplot as plt  
import seaborn as sns                                       # Make charting easier.
import sklearn                                              # Train and evaluate machine learning models.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import category_encoders as ce                              # Encode data.
import warnings                                             # Suppress warnings.
warnings.filterwarnings('ignore')

# Summarize software libraries used.
print('Libraries used in this project:')
print('- Python {}'.format(sys.version))
print('- NumPy {}'.format(np.__version__))
print('- pandas {}'.format(pd.__version__))
print('- SciPy {}'.format(sp.__version__))
print('- Matplotlib {}'.format(matplotlib.__version__))
print('- Seaborn {}'.format(sns.__version__))
print('- scikit-learn {}'.format(sklearn.__version__))
print('- Category Encoders {}'.format(ce.__version__))


# # Load and preview the data

# In[388]:


users_data = pd.read_pickle('data/users_data_cleaned.pickle')

users_data.head(n=5)


# # Check the shape of the data

# In[389]:


users_data.shape


# # Check the number of unique users

# In[390]:


len(np.unique(users_data.user_id))


# # Check the data types

# In[391]:


users_data.info()


# In[392]:


users_data.columns.to_series().groupby(users_data.dtypes).groups


# # Check for correlations

# In[393]:


users_data.corr().abs()


# # Generate summary statistics for all of the data

# In[394]:


users_data.describe(datetime_is_numeric = True, include = 'all')


# # Generate summary statistics for numerical data only

# In[395]:


users_data.describe()


# # Generate modal values for all data

# In[396]:


# Drop user ID since it's unique.

users_data.drop(['user_id'], axis = 1).mode()


# # Generate skewness and kurtosis measurements

# In[397]:


users_data.skew()


# In[398]:


users_data.kurt()


# # Plot histograms for all numerical columns

# In[399]:


users_data_for_hist = users_data.select_dtypes(exclude = ["bool"])


# In[400]:


users_data_for_hist.hist(figsize = (20, 10), alpha = 0.5, 
                         edgecolor = 'black', grid =False);


# # Generate a box plot for `age`

# In[401]:


users_data['age'].describe()


# In[402]:


plt.figure(figsize = (6, 2))
sns.boxplot(x = users_data['age'], linewidth = 0.9);


# # Generate a violin plot for `age`

# In[403]:


sns.violinplot(x = users_data['age'], linewidth = 0.9);


# # Generate a box plot for `number_transactions`

# In[404]:


users_data['number_transactions'].describe()


# In[405]:


plt.figure(figsize = (6, 2))
sns.boxplot(x = users_data['number_transactions'], linewidth = 0.9);


# # Generate a violin plot for `number_transactions`

# In[406]:



sns.violinplot(x = users_data['number_transactions'], linewidth = 0.9);


# # Generate scatter plots comparing `total_amount_usd` to `number_transactions`

# In[407]:


sns.scatterplot(data = users_data, x = 'total_amount_usd', 
                y ='number_transactions');


# In[408]:


sns.scatterplot(data = users_data, x = 'total_amount_usd', 
                y ='number_transactions', hue = 'education');


# # Generate a line plot for `total_amount_usd`

# In[409]:


years = users_data['date_joined'].dt.year

sns.lineplot(data = users_data, x = years,
            y = "total_amount_usd", 
             estimator = np.mean);


# # Generate bar charts for `job`

# In[410]:


users_job_dist = users_data['job'].value_counts(dropna = False)

users_job_dist


# In[411]:


# Vertical bar chart.
users_job_dist.plot(kind = 'bar')
plt.title('Distribution of Jobs');



# In[412]:


# Horizontal bar chart

users_job_dist.plot(kind = 'barh')
plt.title('Distribution of Jobs');


# In[413]:


# Exclude missing values.
users_data['job'].value_counts().plot(kind = 'bar');


# # Generate a bar chart for `marital`

# In[414]:


users_marital_dist = users_data['marital'].value_counts(dropna = False)

users_marital_dist


# In[415]:


users_marital_dist.plot(kind = 'bar')
plt.title('Distribution of Marital Status')


# # Generate a bar chart for `education`

# In[416]:


users_education_dist = users_data['education'].value_counts(dropna = False)

users_education_dist


# In[417]:


users_education_dist.plot(kind = 'bar')
plt.title('Distribution of Education')


# # Generate a bar chart for `contact`

# In[418]:


users_contact_dist = users_data['contact'].value_counts(dropna = False)

users_contact_dist.plot(kind = 'bar')
plt.title('Distribution of Contact')


# # Generate a bar chart for `poutcome`

# In[419]:


users_poutcome_dist = users_data['poutcome'].value_counts(dropna = False)

users_contact_dist.plot(kind = 'bar')
plt.title('Distribution of poutcome')


# # Generate a bar chart for `default`

# In[420]:


users_device_dist = users_data['default'].value_counts(dropna = False)

users_device_dist.plot(kind = 'bar')
plt.title('Distribution of Loan Default')


# # Generate a bar chart for `device`

# In[421]:


users_device_dist = users_data['device'].value_counts(dropna = False)


# In[422]:


users_device_dist.plot(kind = 'bar',
                       alpha = 0.5, edgecolor = 'black')
plt.title('Distribution of the Devices',
          size = 12, weight ='bold')
plt.xticks(rotation = 45, size = 11)
plt.xlabel('Device Name', size = 10, weight = 'bold')
plt.ylabel('Frequency', size = 10, weight = 'bold')
plt.show();


# # Generate a heatmap for the feature correlations

# In[423]:


corr_matrix = users_data.corr()
corr_matrix


# In[424]:


fig = plt.figure(figsize = (10, 7.5))

sns.heatmap(corr_matrix)


# # Format the heatmap to make it easier to read

# In[425]:


fig = plt.figure(figsize = (11, 8))

sns.heatmap(corr_matrix,
            cmap = 'seismic',
            linewidth = 0.75,
            linecolor = 'black',
            cbar = True,
            vmin = -1,
            vmax = 1,
            annot = True,
            annot_kws = {'size': 8, 'color': 'black'})

plt.tick_params(labelsize = 10, rotation = 45)
plt.title('Correlation Plot', size = 14)


# In[ ]:





# # Identify missing values

# In[426]:


users_data.isnull().sum()


# # Identify the percentage of missing values for each feature

# In[427]:


percent_missing = users_data.isnull().mean()

percent_missing


# # Generate a missing value report

# In[428]:


def missing_value_pct_df(data):
    """Create a DataFrame to summarize missing values."""
  
    percent_missing = data.isnull().mean() 
    missing_value_df =     pd.DataFrame(percent_missing).reset_index()
    
    missing_value_df =     missing_value_df.rename(columns = {'index': 'column_name',
                                       0: 'percent_missing'})

    # Multiply by 100 and round to 4 decimal places.
    missing_value_df['percent_missing'] =     missing_value_df['percent_missing'].     apply(lambda x: round(x * 100, 2)) 

    missing_value_df =     missing_value_df.sort_values(by = ['percent_missing'],
                                 ascending = False)

    return missing_value_df


# In[429]:


missing_value_df = missing_value_pct_df(users_data)

missing_value_df


# # Remove features with a high percentage of missing values

# In[430]:


# Threshold above which to drop feature.
threshold = 80

cols_to_drop = list(missing_value_df[missing_value_df['percent_missing']                       >threshold]['column_name'])

print('Number of features to drop:', 
         missing_value_df[\
         missing_value_df['percent_missing'] > threshold].shape[0])

print(f'features with missing values greater than {threshold}%:', cols_to_drop)


# In[431]:


users_data_cleaned = users_data.drop(cols_to_drop, axis = 1)


# In[432]:


# Confirm feature was dropped.
missing_value_df = missing_value_pct_df(users_data_cleaned)

missing_columns =list(missing_value_df[missing_value_df['percent_missing']                      > 0]['column_name'])

print('Number of features with missing values: ', len(missing_columns))


# # Identify numerical data with missing values

# In[433]:


dtypes = ['int64', 'float64']

numerical_columns = list(users_data_cleaned.select_dtypes(dtypes).columns)

print('Numerical features with missing values: ',       list(set(numerical_columns).intersection(missing_columns)))


# # Impute missing data values for `total_amount_usd`

# In[434]:


# Find a sample user with missing value

sample_user = users_data_cleaned[users_data_cleaned['total_amount_usd'].                   isnull()].sample(1).user_id

sample_user


# In[435]:


# Print mean of total_amount_usd.

print('Mean total_amount_usd:', 
       round(users_data_cleaned['total_amount_usd'].mean(), 2))

# Impute missing values for total_amount_usd with mean.
users_data_cleaned['total_amount_usd']. fillna(round(users_data_cleaned['total_amount_usd'].mean(), 2), inplace = True)


# In[436]:


users_data_cleaned[users_data_cleaned.                   user_id.isin(sample_user)]['total_amount_usd']


# # Replace missing values for `number_transactions` with `0`

# In[437]:


users_data_cleaned['number_transactions']. fillna(0, inplace = True)


# In[438]:


users_data_cleaned[users_data_cleaned.                   user_id.isin(sample_user)]['number_transactions']


# # Identify categorical data with missing values

# In[439]:


categorical_columns = list(users_data_cleaned.select_dtypes(['object']).columns)

print('Categorical features with missing value:',
     list(set(categorical_columns).intersection(missing_columns)))


# # Replace categorical missing values with `'Unknown'`

# In[440]:


users_data_cleaned.device.fillna('unknown', inplace = True)
users_data_cleaned.education.fillna('unknown', inplace = True)
users_data_cleaned.contact.fillna('unknown', inplace = True)
users_data_cleaned.job.fillna('unknown', inplace = True)


# In[441]:


users_data_cleaned.device.value_counts()


# # Check if there are any other missing values

# In[442]:


missing_value_df = missing_value_pct_df(users_data_cleaned)

missing_columns =list(missing_value_df[missing_value_df['percent_missing']                      > 0]['column_name'])

print('Number of features with missing values: ', len(missing_columns))
print('Features with missing values', missing_columns)


# # Remove all rows where `date_joined` is missing

# In[443]:


print('Number of users with corrupted data:',
       users_data_cleaned[users_data_cleaned['date_joined']. \
       isnull()].shape[0])


# In[444]:


# Remove corrupted data.
users_data_cleaned = users_data_cleaned[~users_data_cleaned['date_joined'].isnull()]


# In[445]:


# Check to see if any corrupted rows remain.
print('Number of users with corrupted data:',
       users_data_cleaned[users_data_cleaned['date_joined']. \
       isnull()].shape[0])


# # Perform one last check for missing values

# In[446]:


missing_value_df = missing_value_pct_df(users_data_cleaned)

missing_columns =list(missing_value_df[missing_value_df['percent_missing']                      > 0]['column_name'])

print('Number of features with missing values: ', len(missing_columns))
print('Features with missing values', missing_columns)


# # View the distribution of `age`

# In[447]:


users_data_cleaned['age'].hist()
plt.title('Original Distribution of Age')


# # Apply a log transformation to `age`

# In[448]:


np.log(users_data_cleaned['age']).hist()
plt.title('Log Transformation of Age')


# # Apply a Box–Cox transformation to `age`

# In[449]:


from scipy import stats

pd.Series(stats.boxcox(users_data_cleaned['age'])[0]).hist()
plt.title('Box-Cox Transformation of Age')


# # Identify categorical features

# In[450]:


categorical_columns = list(users_data_cleaned.select_dtypes(['object']).columns)

print('The number of categorical features:', 
         len(categorical_columns))
print('The names of categorical features:', categorical_columns)






# # One-hot encode `job`

# In[451]:


users_data_cleaned.job.value_counts(dropna = True)


# In[452]:


# Create object for one-hot encoding.
encoder = ce.OneHotEncoder(cols = 'job',
                           return_df = True,
                           use_cat_names = True)


# In[453]:


# Fit and transform data.
users_data_encoded = encoder.fit_transform(users_data_cleaned)

# Preview the data.

users_data_encoded.head()


# In[454]:


list(users_data_encoded)


# In[455]:


print('shape of data before encoding:',
     users_data_cleaned.shape)
print('Shape of data after encoding:',
     users_data_encoded.shape)



# # Dummy encode `marital`

# In[456]:


marital_encoded = pd.get_dummies(data= users_data_encoded['marital'],
              drop_first = True)
marital_encoded


# In[457]:


# Conncatenate the new encoded columns.

users_data_encoded =pd.concat([users_data_encoded, marital_encoded], axis =1)

# Drop the original variable.
users_data_encoded.drop(['marital'], axis = 1, inplace = True)


# Preview the data
users_data_encoded.head()


# In[458]:


print('Shape of data after encoding:',
     users_data_encoded.shape)

list(users_data_encoded)


# # One-hot encode the remaining categorical variables

# In[459]:


cols = ['education', 'contact', 'device']

encoder = ce.OneHotEncoder(cols = cols,
                           return_df = True,
                           use_cat_names = True)


# In[460]:


# Fit and transform data.
users_data_encoded = encoder.fit_transform(users_data_encoded)

# Preview the data.
users_data_encoded.head()


# In[461]:


print('Shape of data after encoding:',
     users_data_encoded.shape)

list(users_data_encoded)



# # Discretize `age` into bins

# In[462]:


users_data_encoded.age.describe()


# In[463]:


# Define age bins and labels.

bins = [18, 25, 35, 45, 55, 65, 75, 110]
labels = ['18–24', '25–34', '35–44',
          '45–54', '55–64', '65–74', '75+']

# Perform binning using bin list.
users_data_encoded['age_group'] = pd.cut(users_data_encoded['age'], bins = bins,
      labels = labels, right = False)

# Map bins to integer values.
users_data_encoded['age_group_encoded'] = users_data_encoded['age_group'].cat.codes


# In[464]:


# Verify correct binning.

age_vars = ['age_group_encoded','age_group','age']

users_data_encoded[age_vars].sample(10)


# # Plot the new distribution of `age`

# In[465]:


user_age_dist = users_data_encoded.age_group.value_counts()

user_age_dist


# In[466]:


user_age_dist.plot(kind = 'bar', edgecolor = 'black')
plt.title('Distribution of Age Category')


# In[467]:


# Check against encoded values.
users_data_encoded.age_group_encoded.value_counts().plot(kind = 'bar')


# # Drop the `age` and `age_group` variables

# In[468]:


users_data_encoded.drop(['age', 'age_group'], axis = 1, inplace = True)

list(users_data_encoded)


# # Create a `month_joined` variable from `date_joined`

# In[469]:


users_data_encoded['month_joined'] = users_data_encoded.date_joined.dt.month


# In[470]:


# View the distribution of data.
users_data_encoded['month_joined'].hist()
plt.title('Distribution of Months the users Joined')


# In[471]:


users_data_encoded.drop(['date_joined'],
                       axis = 1 , inplace = True)

list(users_data_encoded)


# # Remove features with low variance

# In[472]:


users_data_encoded.std()


# In[ ]:





# In[473]:


# Define standard deviation threshold.
threshold = 0.1

# Identify features below threshold.
cols_to_drop = list(users_data_encoded.std()[users_data_encoded.std()                               <threshold].index.values)
print('Features with low standard deviaion:', cols_to_drop)


# In[474]:


# Drop features below threshold.

users_data_interim = users_data_encoded.drop(cols_to_drop, axis = 1)


list(users_data_interim)


# # Drop highly correlated features

# In[475]:


# Define correlation threshold.

threshhold = 0.75

corr_matrix = users_data_encoded.corr().abs()
high_corr_var = np.where(corr_matrix >= threshold)
high_corr_var = [(corr_matrix.index[x],
                  corr_matrix.columns[y],
                  round(corr_matrix.iloc[x, y], 2))
                for x, y in zip(*high_corr_var)
                 if x != y and x < y]
high_corr_var


# In[476]:


# Tidy up the output.

record_collinear = pd.DataFrame(high_corr_var).rename(columns = {0: 'drop_feature',
                  1: 'corr_feature',
                  2: 'corr_values'})

record_collinear = record_collinear. sort_values(by = 'corr_values', ascending = False)

record_collinear = record_collinear.reset_index(drop = True)

record_collinear


# In[477]:


cols_to_drop = list(record_collinear['drop_feature'])
print(cols_to_drop)


# In[478]:


users_data_final = users_data_interim.drop(cols_to_drop, errors='ignore')

list(users_data_final)


# # Filter by demographics data

# In[479]:


users_data_demographics = users_data_final.filter(regex = 'education|job|age|single')

users_data_demographics.head(n=3)


# # Standardize the demographics data

# In[480]:


users_data_demographics.describe()


# In[481]:


scaler = StandardScaler()

scaler.fit(users_data_demographics)
users_data_scaled = scaler.transform(users_data_demographics)

print('New standard deviation: ', users_data_scaled.std())
print('New mean:         ', round(users_data_scaled.mean()))


# # Perform PCA to reduce the dimensionality of the demographics dataset

# In[482]:


pca = PCA(n_components = 2, random_state = 1)

pca.fit(users_data_scaled)

reduced = pca.transform(users_data_scaled)


# In[484]:


reduced_df = pd.DataFrame(reduced, columns = ['PCA1','PCA2'])
reduced_df


# # Load the final dataset

# In[485]:


users_data_final.info()


# In[486]:


users_data_final.to_pickle('users_data_final.pickle')


# # Load the demographics dataset with PCA applied

# In[488]:


reduced_df.to_pickle('users_data_demo_pca.pickle')

