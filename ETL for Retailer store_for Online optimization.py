#!/usr/bin/env python
# coding: utf-8

# # Import software libraries

# In[2]:


import sys           # Read system parameters.
import pandas as pd  # Manipulate and analyze data.
import sqlite3       # Manage SQL databases.

# Summarize software libraries used.
print('Libraries used in this project:')
print('- Python {}'.format(sys.version))
print('- pandas {}'.format(pd.__version__))
print('- sqlite3 {}'.format(sqlite3.sqlite_version))


# # Load a CSV file as a `DataFrame`

# In[3]:


complaints_data = pd.read_csv('data/consumer_loan_complaints.csv')


# # Preview the first three rows of the data

# In[4]:


print(complaints_data.head(3))


# # Create a connection to the SQLite database

# In[5]:


conn = sqlite3.connect('data/user_data.db')
conn


# # Read the `users` data

# In[6]:


# Write a query that selects everything from the users table.

query = 'SELECT * FROM users'


# In[7]:


# Read the query into a DataFrame.

users = pd.read_sql(query, conn)

# Preview the data.

users.head()


# In[8]:


# Check the shape of the data.

users.shape


# # Read the `device` data

# In[9]:


query = 'SELECT * FROM device'

device = pd.read_sql(query, conn)

device.head()


# In[10]:


device.shape


# # Read the `transactions` data

# In[11]:


# Read the user transactions in the last 30 days. 

query = 'SELECT * FROM transactions'

transactions = pd.read_sql(query, conn)

transactions.head()


# In[12]:


transactions.shape


# # Aggregate the `transactions` data

# In[13]:


# Aggregate data on the number of transactions and the total amount.

query = '''SELECT user_id,
                  COUNT(*) AS number_transactions,
                  SUM(amount_usd) AS total_amount_usd
           FROM transactions
           GROUP BY user_id'''
transactions_agg = pd.read_sql(query, conn)
transactions_agg.head()


# In[14]:


transactions_agg.shape


# # Merge the `device` table with the `users` table

# In[15]:


# Do a left join, as all users in the users table are of interest.

query = '''SELECT left_table.*,
                  right_table.device
           FROM users AS left_table
           LEFT JOIN device AS right_table 
             ON left_table.user_id = right_table.user_id'''

users_w_device = pd.read_sql(query, conn)


# In[16]:


users_w_device.head(3)


# In[17]:


users_w_device.shape


# # Close the database connection

# In[18]:


conn.close()


# # Merge `users_w_device` with `transactions_agg`

# In[19]:


# Do a right join so users won't be lost.

users_w_devices_and_transactions = transactions_agg.merge(users_w_device, 
                        on = 'user_id', how = 'right')

users_w_devices_and_transactions.head()


# In[20]:


# Make sure number of rows is equal to users_w_devices table.

users_w_devices_and_transactions.shape


# # Identify data where `age` is greater than 150

# In[21]:


users_w_devices_and_transactions[users_w_devices_and_transactions.age > 150]


# # Drop incorrect data

# In[22]:


users_cleaned = users_w_devices_and_transactions[users_w_devices_and_transactions.age <150]

users_cleaned.shape



# # Identify more potentially erroneous data

# In[23]:


# Compare age to device.

pd.crosstab(users_cleaned['age'], users_cleaned['device'])


# # Identify data types that need correcting

# In[24]:


users_cleaned.info()


# In[25]:


users_cleaned.default.value_counts()


# # Convert the relevant variables to a Boolean type

# In[26]:


users_cleaned_1 = users_cleaned.copy()

users_cleaned_1.default =users_cleaned_1.default.map(dict(yes = 1, no = 0)).astype(bool)

users_cleaned_1.default.value_counts()


# In[27]:


# Do the same for the other Boolean variables.

bool_vars = ['housing', 'loan', 'term_deposit']

for var in bool_vars:
    users_cleaned_1[var]=     users_cleaned_1[var].map(dict(yes = 1, no = 0)).astype(bool)
    
    print(f'converted {var} to Boolean')


# In[28]:


users_cleaned_1.info()


# # Convert `date_joined` to a datetime format

# In[35]:


# Work with a new object.
users_cleaned_2 = users_cleaned_1.copy()

users_cleaned_2['date_joined'] = pd.to_datetime(users_cleaned_2['date_joined'], format ='%Y-%m-%d')


# In[36]:


users_cleaned_2.info()


# # Identify all duplicated data

# In[39]:


duplicated_data =users_cleaned_2[users_cleaned_2.duplicated(keep = False)]

print('Number of rows with duplication data:', duplicated_data.shape[0])


# In[40]:


duplicated_data


# # Remove the duplicated data

# In[44]:


users_cleaned_final =users_cleaned_2[~users_cleaned_2.duplicated()]

users_cleaned_final[users_cleaned_final['user_id']=='cba59442-af3c-41d7-a39c-0f9bffba0660']


# In[45]:


users_cleaned_final.shape 


# # Load data into an SQL database

# In[47]:


conn = sqlite3.connect('users_data_cleaned.db')

users_cleaned_final.to_sql('users_cleaned_final',conn,
                           if_exists ='replace', index =False)


# # Confirm that data was loaded into the database

# In[49]:


query = 'SELECT * FROM users_cleaned_final'

pd.read_sql(query, conn).head()


# # Close the database connection

# In[50]:


conn.close()


# # Write the `DataFrame` as a pickle file

# In[51]:


users_cleaned_final.to_pickle("users_data_cleaned.pickle")


# # Confirm that the data was written to the pickle file

# In[52]:


pd.read_pickle('users_data_cleaned.pickle').head()


# In[53]:


pd.read_pickle('users_data_cleaned.pickle').info()


# # Write the data to a CSV file

# In[54]:


users_cleaned_final.to_csv('users_data_cleaned.csv', index = False)


# # Confirm that the data was written to a CSV file.

# In[55]:


pd.read_csv('users_data_cleaned.csv').head()


# In[56]:


pd.read_csv('users_data_cleaned.csv').info()

