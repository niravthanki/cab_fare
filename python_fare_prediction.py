

# load required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 16
palette = sns.color_palette('Paired', 10)


# set working directory
os.chdir("D:/Project_1")

# read train & test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Shape of Train Data:", train.shape)
print("Shape of Test Data:", test.shape)


train.head()
test.head()

# check data type of Train dataset
train.dtypes

# check data type of Test dataset
test.dtypes


##############################
#                            #
#   Missing Value Analysis   #
#                            #
##############################

train.isnull().sum().sort_values(ascending = False)

# print("Length of Train Data before removing Missing value:", len(train))
train = train.dropna(how = 'any', axis = 0)
print("Length of Train Data after removing Missing value:", len(train))


train.info()


################################
#                              #
#   Exploratory data analysi   #
#                              #
################################


##################################
#                                #
#  Explore variable fare_amount  #
#                                #
##################################

# change the data type of fare_amount from object to float
train['fare_amount'] = pd.to_numeric(train['fare_amount'], errors = 'coerce')
print(f"Data type of fare_amount:{train['fare_amount'].dtype}")

# checking missing value in Training Dataset after convert 'fare_amount' from object to float
train.isnull().sum().sort_values(ascending = False)

# print("Length of Train Data before removing Missing value:", len(train))
train = train.dropna(how = 'any', axis = 0)
print("Length of Train Data after removing Missing value:", len(train))

# summary of fare amount
train['fare_amount'].describe()

# Histogram of Fare Amount
train.fare_amount.hist(bins = 50, figsize = (14,4))
plt.xlabel('Fare Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Fare Amount')

train.shape

print(f"There are {len(train[train['fare_amount']<0])} observation where fare amount is less than 0")
print(f"There are {len(train[train['fare_amount']==0])} observation where fare amount is equal to 0")
print(f"There are {len(train[train['fare_amount']<2.5])} observation where fare amount is less than 2.5")
print(f"There are {len(train[train['fare_amount']>100])} observation where fare amount is greater than 100")
print(f"There are {len(train[train['fare_amount']>500])} observation where fare amount is greater than 500")


# define function 'clean_fare' to remove fare which is less than 2.5 and greate than 100
def clean_fare(df):
    return df[(df.fare_amount >= 2.5)  & (df.fare_amount <= 100)]

print("Length of Train data set before removing outlier from fare_amount:", len(train))
train = clean_fare(train)
print("Length of Train data set after removing outlier from fare_amount:", len(train))

train.fare_amount.hist(bins = 50, figsize = (14,4))
plt.xlabel('Fare Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Fare Amount after removing outliers')

#################################
#                               #
#   Explore Location variables  #
#                               #
#################################

# summary of train data for distance related variables
train[['pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude']].describe()

# summary of test data for distance related variables
test[['pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude']].describe()


# Range of Pickup_longitude & Dropoff_longitude as per Test Data
min(test.pickup_longitude.min(), test.dropoff_longitude.min()),max(test.pickup_longitude.max(), test.dropoff_longitude.max())


# Range of Pickup_latitude & Dropoff_latitude as per Test Data
min(test.pickup_latitude.min(), test.dropoff_latitude.max()),max(test.pickup_latitude.max(), test.dropoff_latitude.max())


# define function 'clean_lon_lat' to set training data in range with test data
def clean_lon_lan(df, TB):
    return(df.pickup_longitude >= TB[0]) & (df.pickup_longitude <= TB[1]) &
            (df.pickup_latitude >= TB[2]) & (df.pickup_latitude <= TB[3]) &
            (df.dropoff_longitude >= TB[0]) & (df.dropoff_longitude <= TB[1]) &
            (df.dropoff_latitude >= TB[2]) & (df.dropoff_latitude <= TB[3])

TB = (-74.3, -73, 40.6, 41.8)
print('Length of Train Data before removing outlier in Longitude & Latitude:', len(train))
train = train[clean_lon_lan(train,TB)]
print('Length of Train Data after removing outlier in Longitude & Latitude::', len(train))

# Scatter plot of Pickup Location
plt.figure(figsize=(14,5))
plt.scatter(x=train['pickup_latitude'], y=train['pickup_longitude'], color = "b")
plt.xlabel('Pickup Latitude')
plt.ylabel('Pickup Longitude')
plt.title('Pickup Location')


# Scatter plot of Dropoff Location
plt.figure(figsize=(14,5))
plt.scatter(x=train['dropoff_latitude'], y=train['dropoff_longitude'], color = "b")
plt.xlabel('Dropoff Latitude')
plt.ylabel('Dropoff Longitude')
plt.title('Dropoff Location')



######################################
#                                    #
#  Explore variable passenger_count  #
#                                    #
######################################

# change the data type of passenger_count from float to integer
train['passenger_count'] = train['passenger_count'].astype(float).round().astype(np.int64)
print(f"Data type of fare_amount:{train['passenger_count'].dtype}")

# summary of variable 'passenger_count' as per Train data
train['passenger_count'].describe()

# summary of variable 'passenger_count' as per Train data
test['passenger_count'].describe()

print("Below details of Test Dataset")
print(f"There are {len(test[test['passenger_count']==0])} observation where Passenger count is equal to 0.")
print(f"There are {len(test[test['passenger_count']<6])} obseravtion where Passenger count is Less than 6.")
print(f"There are {len(test[test['passenger_count']==6])} observation where Passenger count is equal to 6.")


print("Below details of Train Dataset")
print(f"There are {len(train[train['passenger_count']==0])} observation where Passenger count is equal to 0.")
print(f"There are {len(train[train['passenger_count']<6])} obseravtion where Passenger count is Less than 6.")
print(f"There are {len(train[train['passenger_count']==6])} observation where Passenger count is equal to 6.")
print(f"There are {len(train[train['passenger_count']>6])} obseravtion where Passenger count is Greater than 6.")


# Histogram of passenger count with outliers
train['passenger_count'].value_counts().sort_index().plot.bar(color = 'b', figsize = (14,4))
plt.xlabel('Passenger Count')
plt.ylabel('Frequency')
plt.title('Histogram of Passenger count with outliers')


# define function 'clean_pass' to remove passenger count which is greater than 6
def clean_pass(df):
    return df[(df.passenger_count > 0) & (df.passenger_count <= 6)]


print("Length of Train Data before removing outlier in Passenger Count:", len(train))
train = clean_pass(train)
print("Length of Train Data after removing outlier in Passenger Count:", len(train))

# checking correlatin between passenger count and target variable fare amount
train['passenger_count'].corr(train['fare_amount'])

# Histogram of passenger count after removing outliers
train['passenger_count'].value_counts().sort_index().plot.bar(color = 'b', figsize = (14,4))
plt.xlabel('Passenger Count')
plt.ylabel('Frequency')
plt.title('Histogram of Passenger Count without outliers')




######################################
#                                    #
#  Explore variable pickup_datetime  #
#                                    #
######################################

train['pickup_datetime'].describe()
test['pickup_datetime'].describe()

# convert 'pickup_datetime' into pandas date time format in Train Dataset
train['pickup_datetime'] = train['pickup_datetime'].str.slice(0,16)
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], utc = True, format = '%Y-%m-%d %H:%M', errors = 'coerce')

# convert 'pickup_datetime' into pandas date time format in Test Dataset
test['pickup_datetime'] = test['pickup_datetime'].str.slice(0,16)
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], utc = True, format = '%Y-%m-%d %H:%M', errors = 'coerce')

print("Type of pickup_datetime after conversaion in Train data:",train['pickup_datetime'].dtypes)
print("Type of pickup_datetime after conversaion in Test data:",test['pickup_datetime'].dtypes)


# before moving further let's check once again is there any NA or not? in Train data
train.isnull().sum().sort_values(ascending = False)

print("Length of Train Data before removing NA:", len(train))
train = train.dropna(how = 'any', axis = 0)
print("Length of Train Data after removing NA:", len(train))


#########################
#                       #
#  Feature Engineering  #
#                       #
#########################


####################################
#                                  #
#        create new features       #
#  'abs_lat_diff' & 'abs_lon_diff  #
#                                  #
####################################

# define function 'absolute' to compute absolute difference between pickup & dropoff cordinaters 
def absolute(df):
    df['abs_lat_diff'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()
    df['abs_lon_diff'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()
    
    return df


# create new variables in Train & Test data
train = absolute(train)
test = absolute(test)

print("Correlation between abs_lat_diff & fare_amount is",round(train['abs_lat_diff'].corr(train['fare_amount']),2))
print("Correlation between abs_lon_diff & fare_amount is",round(train['abs_lon_diff'].corr(train['fare_amount']),2))



############################
#                          #
#    create new feature    #
#        'distance'        #
#                          #
############################

# define function to Compute Haversine distance
def haversine_dis(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
   
    #Define earth radius (km)
    R_earth = 6371
    
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    km = R_earth * c
    
    return km


# create variable 'distance' for Train & Test dataset
train['distance'] = haversine_dis(train['pickup_latitude'], train['pickup_longitude'], 
                                   train['dropoff_latitude'], train['dropoff_longitude']) 
test['distance'] = haversine_dis(test['pickup_latitude'], test['pickup_longitude'],
                                test['dropoff_latitude'], test['dropoff_longitude'])

print("Correlation between Distance and Fare Amount in Train Data:",round(train['distance'].corr(train['fare_amount']),2))


# scatter plot of distance & Fare amount
plt.figure(figsize = (14,7))
plt.scatter(x = train['distance'], y = train['fare_amount'], color = "b")
plt.xlabel('Distance')
plt.ylabel('Fare Amount')
plt.title('Scatter Plot of Distance & Fare Amount')



###########################
#                         #
#   create time series    #
#        features         #
#                         #
###########################

# define function create_date_time for variable 'pickup_datetime'
def create_date_time(df):
    df['hour_of_day'] = df.pickup_datetime.dt.hour
    df['month'] = df.pickup_datetime.dt.month
    df['year'] = df.pickup_datetime.dt.year
    df['day'] = df.pickup_datetime.dt.day
    df['weekday'] = df.pickup_datetime.dt.weekday
    return df


# create new variables in Train & Test data for Date & Time
train = create_date_time(train)
test = create_date_time(test)


# histogram of hour_of_day to check frequency of rides during the day
plt.figure(figsize=(14,5))
train['hour_of_day'].value_counts().sort_index().plot.bar(color = "b")
plt.xlabel('Hour of Day')
plt.ylabel('Frequency')
plt.title('Bar Graph of Hour of Day')


# scatter plot of 'hour_of_day' & 'fare_amount' 
plt.figure(figsize=(14,5))
plt.scatter(x=train['hour_of_day'], y=train['fare_amount'],color = "b")
plt.xlabel('Hour of the Day')
plt.ylabel('Fare')
plt.title('Scatter Plot of Hour of day & Fare Amount')


# # histogram of weekday to check frequency of rides during the week
plt.figure(figsize = (14,5))
train['weekday'].value_counts().sort_index().plot.bar(color = "b")
plt.xticks([0,1,2,3,4,5,6],['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.ylabel('Frequency')
plt.title('Bar Graph of weekday')




####################################
#                                  #
#   create new features base on    #
#        popular locations         #
#                                  #
####################################

def create_airport_dis(df):
    
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    nyc_coord = (40.7141667,-74.0063889) 
    
    
    pickup_lat = df['pickup_latitude']
    dropoff_lat = df['dropoff_latitude']
    pickup_lon = df['pickup_longitude']
    dropoff_lon = df['dropoff_longitude']
    
    pickup_jfk = haversine_dis(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = haversine_dis(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = haversine_dis(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = haversine_dis(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = haversine_dis(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = haversine_dis(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)
    pickup_sol = haversine_dis(pickup_lat, pickup_lon, sol_coord[0], sol_coord[1]) 
    dropoff_sol = haversine_dis(sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon)
    pickup_nyc = haversine_dis(pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1]) 
    dropoff_nyc = haversine_dis(nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon)
    
    
    
    df['jfk_dist'] = pickup_jfk + dropoff_jfk
    df['ewr_dist'] = pickup_ewr + dropoff_ewr
    df['lga_dist'] = pickup_lga + dropoff_lga
    df['sol_dist'] = pickup_sol + dropoff_sol
    df['nyc_dist'] = pickup_nyc + dropoff_nyc
    
    return df


train = create_airport_dis(train)
test = create_airport_dis(test)

# check correlation with fare amount
print("Correlation between jfk_dist & fare_amount is",round(train['jfk_dist'].corr(train['fare_amount']),2))
print("Correlation between ewr_dist & fare_amount is",round(train['ewr_dist'].corr(train['fare_amount']),2))
print("Correlation between lga_dist & fare_amount is",round(train['lga_dist'].corr(train['fare_amount']),2))
print("Correlation between sol_dist & fare_amount is",round(train['sol_dist'].corr(train['fare_amount']),2))
print("Correlation between nyc_dist & fare_amount is",round(train['nyc_dist'].corr(train['fare_amount']),2))


# check correlation with target variable
corr = train.corr()
plt.figure(figsize = (14,5))
corr['fare_amount'].plot.bar(color = "b")
plt.title("Correlation with Target Variable")


########################################
#                                      #
#   split data to Train & validation   #
#                                      #
########################################


# remove 'pickup_datetime' variables from train & test dataset as save for later use of submission
train_date = train['pickup_datetime']
test_date = test['pickup_datetime']

y = train['fare_amount']
y = np.array(y)

x = train.drop(columns = ['pickup_datetime', 'fare_amount'])
test = test.drop(columns = ['pickup_datetime'])


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=123)

print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)


#####################################
#                                   #
#   Evaluation matrix RMSE & MAPE   #
#                                   #
#####################################

from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)


def rmse_metrics(train_pred, valid_pred, y_train, y_valid):
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
      
    return train_rmse, valid_rmse


def mape_metrics(train_pred, valid_pred, y_train, y_valid):
    
     # Calculate absolute percentage error
    train_ape = abs((y_train - train_pred) / y_train)
    valid_ape = abs((y_valid - valid_pred) / y_valid)
    
    # Account for y values of 0
    train_ape[train_ape == np.inf] = 0
    train_ape[train_ape == -np.inf] = 0
    valid_ape[valid_ape == np.inf] = 0
    valid_ape[valid_ape == -np.inf] = 0
    
    train_mape = 100 * np.mean(train_ape)
    valid_mape = 100 * np.mean(valid_ape)
    
    return train_mape, valid_mape



###################################
#                                 #
#   Multiple Linear Regression    #
#        with all variables       #
#                                 #
###################################

# load library for Linear regression model
import statsmodels.api as sm

# create Multiple Linear Regression model with all features
lr = sm.OLS(y_train, X_train).fit()

# summary of model 'lr'
print(lr.summary())

# make prediction of Multiple Linear regression Model for Train & Validation data
lr_train_pred = lr.predict(X_train)
lr_valid_pred = lr.predict(X_valid)

# compute RMSE & MAPE errors for both Training & Validation dataset
lr_tr, lr_vr = rmse_metrics(lr_train_pred, lr_valid_pred, y_train, y_valid)
lr_tm, lr_vm = mape_metrics(lr_train_pred, lr_valid_pred, y_train, y_valid)

print("Multiple Linear Regression RMSE & MAPE score using all features")
print(f'Linear Regression Training:   RMSE = {round(lr_tr, 2)} \t MAPE = {round(lr_tm, 2)}')
print(f'Linear Regression Validation: RMSE = {round(lr_vr, 2)} \t MAPE = {round(lr_vm, 2)}')


# make prediction of Multiple Linear Regression Model 1 for Final Test Dataset
lr_test_pred = lr.predict(test)

# Density Plot of Multiple Linear Regression 
plt.figure(figsize = (10, 6))
sns.kdeplot(y_valid, label = 'Actual')
sns.kdeplot(lr_valid_pred, label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of Valid Fares: Multiple Linear Regression");



###################################
#                                 #
#   Multiple Linear Regression    #
#     with selected features      #
#                                 #
###################################

# list of variables which p value is less than 0.05 
less_pvalue_var = ['pickup_longitude', 'pickup_latitude', 'month', 'year', 'distance',
                              'jfk_dist', 'ewr_dist', 'sol_dist', 'nyc_dist', 'abs_lat_diff',
                              'abs_lon_diff']

# create Multiple Linear Regression model with selected features
lr1 = sm.OLS(y_train, X_train[less_pvalue_var]).fit()

# summary of model 'lr1'
print(lr1.summary())

# make prediction of Multiple Linear Regression model for Train & Validation data
lr1_train_pred = lr1.predict(X_train[less_pvalue_var])
lr1_valid_pred = lr1.predict(X_valid[less_pvalue_var])

# compute RMSE & MAPE errors for both Training & Validation dataset
lr1_tr, lr1_vr = rmse_metrics(lr1_train_pred, lr1_valid_pred, y_train, y_valid)
lr1_tm, lr1_vm = mape_metrics(lr1_train_pred, lr1_valid_pred, y_train, y_valid)

print("Multiple Linear Regression RMSE & MAPE score using selected features")
print(f'Linear Regression Training:   RMSE = {round(lr1_tr, 2)} \t MAPE = {round(lr1_tm, 2)}')
print(f'Linear Regression Validation: RMSE = {round(lr1_vr, 2)} \t MAPE = {round(lr1_vm, 2)}')

# make prediction of Multiple Linear Regression Model 2 for Final Test Dataset
lr1_test_pred = lr1.predict(test[less_pvalue_var])

# check correlation of target variable with predictors
corrs = train.corr()
plt.figure(figsize = (14,6))
corrs['fare_amount'].plot.bar(color = 'b');
plt.title('Correlation with Fare Amount');



#####################
#                   #
#   Random Forest   #     
#                   #               
#####################

# use Random Forest with all the features and without any hyper parameter tuning
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)

# make prediction for Train & Validation data 
rf_train_pred = rf.predict(X_train)
rf_valid_pred = rf.predict(X_valid)

# compute RMSE & MAPE errors for both Training & Validation dataset
rf_tr, rf_vr = rmse_metrics(rf_train_pred, rf_valid_pred, y_train, y_valid)
rf_tm, rf_vm = mape_metrics(rf_train_pred, rf_valid_pred, y_train, y_valid)

print("Random Forest RMSE & MAPE score using all features")
print(f'Random Forest Training:   RMSE = {round(rf_tr, 2)} \t MAPE = {round(rf_tm, 2)}')
print(f'Random Forest Validation: RMSE = {round(rf_vr, 2)} \t MAPE = {round(rf_vm, 2)}')

# make prediction of Random Forest Default for Final Test Dataset
rf_test_pred = rf.predict(test)

# feature Importance as per Random Forest
feat_importances = pd.Series(rf.feature_importances_, index= X_train.columns)
plt.figure(figsize= (14,5))
feat_importances.sort_index().plot.bar(color = "b")
plt.ylabel("Relative Feature Importance")
plt.title("Featur Importance")

#######################################
#                                     #
#        Random Forest with           #
#  cross-Validation & with 30 trees   #
#                                     #
#######################################

# Random Forest hyper parameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
scorer = metrics.make_scorer(metrics.mean_squared_error)
clf = RandomForestRegressor()

parameters = {'n_estimators': [30],'random_state':[0]}

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# use Random Forest with all variables and with hyper parameter tuning
grid_fit = grid_obj.fit(X_train, y_train)
best_clf = grid_fit.best_estimator_

# make prediction for train and validation data
rf1 = best_clf.predict(X_train)
rf2 = best_clf.predict(X_valid)

# compute RMSE & MAPE errors for both Training & Validation dataset
rf_tr1, rf_vr1 = rmse_metrics(rf1, rf2, y_train, y_valid)
rf_tm1, rf_vm1 = mape_metrics(rf1, rf2, y_train, y_valid)

print("Random Forest RMSE & MAPE score using all features with hyper parameter tuning")
print(f'Random Forest Training:   RMSE = {round(rf_tr1, 2)} \t MAPE = {round(rf_tm1, 2)}')
print(f'Random Forest Validation: RMSE = {round(rf_vr1, 2)} \t MAPE = {round(rf_vm1, 2)}')

# make prediction of Random Forest Default for Final Test Dataset
rf1_test_pred = best_clf.predict(test)

# Density plot of Random Forest with Hyper Parameter and grid search
plt.figure(figsize = (10, 6))
sns.kdeplot(y_valid, label = 'Actual')
sns.kdeplot(rf2, label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of Validation Fares: Random Forest Grid")



###############
#             #      
#   XGBoost   #     
#             #                    
###############

# XGBoost hyper parameter tuning
import xgboost as xgb
def XGBoost(X_train,X_valid,y_train,y_valid):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_valid,label=y_valid)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=400, 
                    early_stopping_rounds=30,evals=[(dtest,'test')])


# XGBoost model
xgbm = XGBoost(X_train,X_valid,y_train,y_valid)

# make prediction for train and validation data
xgb_train_pred = xgbm.predict(xgb.DMatrix(X_train), ntree_limit = xgbm.best_ntree_limit)
xgb_valid_pred = xgbm.predict(xgb.DMatrix(X_valid), ntree_limit = xgbm.best_ntree_limit)

# compute RMSE & MAPE errors for both Training & Validation dataset
xgb_tr, xgb_vr = rmse_metrics(xgb_train_pred, xgb_valid_pred, y_train, y_valid)
xgb_tm, xgb_vm = mape_metrics(xgb_train_pred, xgb_valid_pred, y_train, y_valid)

print("XGBoost RMSE & MAPE score using all features")
print(f'XGBoost Training:   RMSE = {round(xgb_tr, 2)} \t MAPE = {round(xgb_tm, 2)}')
print(f'XGBoost Validation: RMSE = {round(xgb_vr, 2)} \t MAPE = {round(xgb_vm, 2)}')

# make prediction of XGBoost for Final Test Dataset
xgb_test_pred = xgbm.predict(xgb.DMatrix(test), ntree_limit = xgbm.best_ntree_limit)

# Density plot of XGBoost
plt.figure(figsize = (10, 6))
plt.figure(figsize = (10, 6))
sns.kdeplot(y_valid, label = 'Actual')
sns.kdeplot(xgb_valid_pred, label = 'Predicted')
plt.legend(prop = {'size': 30})
plt.title("Distribution of Validation Fares: XGBoost");



###################################
#                                 #
#    Compare the performance of   #
#         all used models         #
#                                 #
###################################

# make data frame 'all_model' for store RMSE score of all used model
all_model = pd.DataFrame({"model": ['Multiple Linear Regression 1','Multiple Linear Regression 2',
                                    'Random Forest 1','Random Forest 2','XGBoost'], 
                           "train_rmse": [lr_tr, lr1_tr, rf_tr, rf_tr1, xgb_tr],
                         "valid_rmse": [lr_vr, lr1_vr, rf_vr, rf_vr1, xgb_vr]},
                         columns = ['model','train_rmse','valid_rmse'])
all_model = all_model.sort_values(by = 'valid_rmse', ascending = False)

# Bar graph of model performance
sns.barplot(all_model['valid_rmse'], all_model['model'], palette = 'Set2')
plt.xlabel('Root Mean Square Error')
plt.ylabel('Models')
plt.title('Compare performance of all used model')

print(all_model)




#################################
#                               #
#   Submission result of all    #
#         used models           #
#                               #
#################################

# add Pickup_datetime which was removed while modeling
test['pickup_datetime'] = test_date


# sumission of Linear Regression 1
test['fare_amount_linear_1'] = lr_test_pred

# submission of Linear Regression 2
test['fare_amount_linear_2'] = lr1_test_pred

# Submission of Random Forest 1
test['fare_amount_random_for_1'] = rf_test_pred

# submission of Random Forest 2
test['fare_amount_random_for_2'] = rf1_test_pred

# submission of XGBoost
test['fare_amount_xgboost'] = xgb_test_pred

# write file to working directory
test.to_csv('Python_sub_all_models.csv', index = False)

