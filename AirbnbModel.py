import pandas as pd
import numpy as np
from numpy.random import seed
import pickle
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import time
from keras import models, layers
seed(123)

raw_df = pd.read_csv('updatedFinalData.csv')
print(f"The dataset contains {len(raw_df)} Airbnb listings")
pd.set_option('display.max_columns', len(raw_df.columns))  # To view all columns
pd.set_option('display.max_rows', 100)
raw_df.head(3)


cols_to_drop = ['Zipcode', 'Name', 'AirbnbExperiences', 'AirbnbHostResponse', 'HostResponseTime', 'hasKba', 'hasLinkedIn', 'hasFacebook', 'hasReviews', 'hasPhone', 'hasEmail', 'Neighbourhood Cleansed', 'State', 'Market',
                # 'Country',
                'Country Code',
                'City',
                # 'Latitude',
                # 'Longitude',
                'Guests Included', 'Extra People', 'Minimum Nights', 'Maximum Nights',
                # 'Number of Reviews',
                'Review Scores Rating',
                'Review Scores Accuracy', 'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication', 'Review Scores Location', 'Review Scores Value', 'hasInstantBookable', 'hasExactLocation', 'hasProfilePic']
df = raw_df.drop(cols_to_drop, axis=1)


# In[7]:


df.isna().sum()
df.set_index('ID', inplace=True)


# In[8]:


# Replacing columns with f/t with 0/1
df.replace({'f': 0, 't': 1}, inplace=True)

# Plotting the distribution of numerical and boolean categories
df.hist(figsize=(20, 20))


# In[9]:

# df.Zipcode.fillna("unknown", inplace=True)
# df.Zipcode.value_counts(normalize=True)


# In[10]:


df.columns = [c.replace(' ', '_') for c in df.columns]
df.RefinedPropertyType.value_counts()


# In[11]:


for col in ['Bathrooms', 'Bedrooms', 'Beds']:
    df[col].fillna(df[col].median(), inplace=True)


# In[12]:

# df.Price = df.Price.str[1:-3]
# df.Price = df.Price.str.replace(",", "")
# df.Price = df.Price.astype('int64')
df.dropna(subset=['Price'], inplace=True)


# In[13]:


df.Cancellation_Policy.value_counts()


# In[14]:


df.Cancellation_Policy.replace({
    'super_strict_30': 'strict',
    'super_strict_60': 'strict'}, inplace=True)


# In[15]:


df.Price.isna().sum()


# In[16]:


transformed_df = pd.get_dummies(df)


# In[17]:


numerical_columns = ['Accommodates', 'Bedrooms', 'Bathrooms', 'Beds']
for col in transformed_df.columns:
    transformed_df[col] = transformed_df[col].astype('float64').replace(0.0, 0.02)  # Replacing 0s with 0.02
    transformed_df[col] = np.log(transformed_df[col])


# In[18]:


# Separating X and y
X = transformed_df.drop('Price', axis=1)
y = transformed_df.Price

# In[19]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=list(X.columns))
X_test = pd.DataFrame(scaler.transform(X_test), columns=list(X.columns))
pickle.dump({"var": scaler.var_, "mean": scaler.mean_}, open('X_train_normalizer.pickle', 'wb'))
transformed_df.shape


# In[30]:


xgb_reg_start = time.time()
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_reg = xgb.XGBRegressor(base_score=0.007, colsample_bylevel=1,
                           colsample_bytree=0.95, gamma=0, learning_rate=0.09,
                           max_delta_step=0, max_depth=11, min_child_weight=1,
                           n_estimators=100, nthread=-1, objective='reg:linear', reg_alpha=0.98,
                           reg_lambda=1, scale_pos_weight=5, seed=0, silent=True,
                           subsample=0.9)
xgb_reg.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
training_preds_xgb_reg = xgb_reg.predict(X_train)
val_preds_xgb_reg = xgb_reg.predict(X_test)

xgb_reg_end = time.time()
results = xgb_reg.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = pyplot.subplots(figsize=(12, 12))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.xlabel('Epochs')
pyplot.title('XGBoost Log Loss')
pyplot.show()

# plot classification error
fig, ax = pyplot.subplots(figsize=(12, 12))
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('Epochs')
pyplot.title('XGBoost Classification Error')
pyplot.show()
print(f"Time taken to run: {round((xgb_reg_end - xgb_reg_start) / 60, 1)} minutes")
print("\nTraining MSE:", round(mean_squared_error(y_train, training_preds_xgb_reg), 4))
print("Test MSE:", round(mean_squared_error(y_test, val_preds_xgb_reg), 4))
print("\nTraining r2:", round(r2_score(y_train, training_preds_xgb_reg), 4))
print("Test r2:", round(r2_score(y_test, val_preds_xgb_reg), 4))
# val_preds_xgb_reg.to_csv('Airbnb-predictions.csv', header=True)
np.savetxt("Airbnb-predictions.csv", np.exp(val_preds_xgb_reg), delimiter=",")
filename = 'Airbnb_model.sav'
pickle.dump(xgb_reg, open(filename, 'wb'))


# In[22]:


ft_weights_xgb_reg = pd.DataFrame(xgb_reg.feature_importances_, columns=['weight'], index=X_train.columns)
ft_weights_xgb_reg.sort_values('weight', inplace=True)
ft_weights_xgb_reg.to_csv('Airbnb-weights.csv', header=True)
ft_weights_xgb_reg


# In[23]:


transformed_df.head(5)
# .to_csv('Airbnb-sample-data',header = True)


# In[24]:


# Building the model
nn2 = models.Sequential()
nn2.add(layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
nn2.add(layers.Dense(256, activation='relu'))
nn2.add(layers.Dense(256, activation='relu'))
nn2.add(layers.Dense(1, activation='linear'))

# Compiling the model
nn2.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['mean_squared_error'])

# Model summary
print(nn2.summary())


# In[25]:


# # Training the model
# nn2_start = time.time()

# nn2_history = nn2.fit(X_train,
#                   y_train,
#                   epochs=100,
#                   batch_size=412,
#                   validation_split = 0.1)

# nn2_end = time.time()
loaded_model = pickle.load(open('Airbnb_model.sav', 'rb'))
xgb.__version__


# In[26]:


# def nn_model_evaluation(model, skip_epochs=0, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):

#     # MSE and r squared values
#     y_test_pred = model.predict(X_test)
#     y_train_pred = model.predict(X_train)
#     print("Training MSE:", round(mean_squared_error(y_train, y_train_pred),4))
#     print("Validation MSE:", round(mean_squared_error(y_test, y_test_pred),4))
#     print("\nTraining r2:", round(r2_score(y_train, y_train_pred),4))
#     print("Validation r2:", round(r2_score(y_test, y_test_pred),4))


# In[27]:


# nn_model_evaluation(nn2)
