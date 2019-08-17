
# load required libraries
library(geosphere) # for calculate haversian distance
library(xgboost) # machine learning
library(caret) # machiner learning
library(randomForest) # machine learning
library(Matrix)
library(scales)
library(caret)
library(dplyr) # data manupilation
library(ggplot2) # data visualization
#devtools::install_github("laresbernardo/lares")
library(lares) # use to plot regression performance
library(ggcorrplot) # for correlation plot
library(corr)

# set working directory
setwd("D:/Project_1")
getwd()


# load train & test data
train = read.csv('train.csv', na.string = c("NA"," ", ""))
test = read.csv('test.csv', na.string = c("NA"," ", ""))

cat("Dimension of Train Data:", dim(train))
cat("Dimension of Test Data:", dim(test))


head(train)
head(test)

# check data type of train & test set
str(train)
str(test)

# summary statistic of data
summary(train)
summary(test)


##############################
#                            #
#   Missing Value Analysis   #
#                            #
##############################

apply(train, 2, function(x){sum(is.na(x))})
apply(test, 2, function(x){sum(is.na(x))})

# remove missing value

cat("Dimension of Train data before removing Missing values", dim(train))
train = na.omit(train)
cat("Dimensiton of Train data after removing Missing values:", dim(train))


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

# check the class of fare_amount 
class(train$fare_amount)

# convert into numeric
train$fare_amount = as.character(train$fare_amount)
train$fare_amount = as.numeric(train$fare_amount)

# checking missing value in Training Dataset after convert 'fare_amount' from object to float
apply(train, 2, function(x){sum(is.na(x))})


cat("Dimension of Train Data before removing missing value:", dim(train))

# remove 'NA' value
train = na.omit(train)

cat("Dimension of Train Data after removing missing value:", dim(train))

# once again check class of 'fare_amount' after conversion
class(train$fare_amount)
str(train$fare_amount)
summary(train$fare_amount)


# Histogram of Fare Amount with outliers
ggplot(train, aes(x = fare_amount))+
  geom_histogram(fill = "blue", bins = 30)+
  ggtitle("Histogram of Fare Amount with outliers")+
  xlab("Fare Amount")+
  ylab("Frequency")


cat("Dimention of Train Data:", dim(train))


cat("There are", nrow(filter(train, fare_amount < 0)), "observation where fare amount is less than 0")
cat("There are", nrow(filter(train, fare_amount == 0)), "observation where fare amount is equal to 0")
cat("There are", nrow(filter(train, fare_amount < 2.5)), "observation where fare amount is less than 2.5")
cat("There are", nrow(filter(train, fare_amount > 100)), "observation where fare amount is greater than 100")
cat("There are", nrow(filter(train, fare_amount > 500)), "observation where fare amount is greater than 500")

cat("Dimention of Train data before removing outlier from fare_amount:", dim(train))


# set fare amount between 0 to 100
train = train %>%
  filter(fare_amount >= 2.5, fare_amount <= 100)

cat("Dimention of Train data after removing outlier from fare_amount:", dim(train))


# Histogram of fare amount after removing outliers
ggplot(aes(x = fare_amount), data = train)+
  geom_histogram(bins = 50, fill = "blue")+
  ggtitle("Histogram of Fare Amount without outliers")+
  xlab("Fare Amount")+
  ylab("Frequency")


#################################
#                               #
#   Explore Location variables  #
#                               #
#################################

# Minimum & Maximum value of Location related varibales in Train Data set
train %>%
  select(pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude) %>%
  summarise_all(list(min = min, max = max))

# Minimum & Maximum value of Location related varibales in Test Data set
test %>%
  select(pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude) %>%
  summarise_all(list(min = min, max = max))

cat("Dimention of Train data before removing outlier from longitude & latitude", dim(train))


# set Train data with range of Test data
train = train %>%
  filter(pickup_latitude >= 40.6 & pickup_latitude <= 41.8) %>%
  filter(pickup_longitude >= -74.3 & pickup_longitude <= -73) %>%
  filter(dropoff_latitude >= 40.6 & dropoff_latitude <= 41.8) %>%
  filter(dropoff_longitude >= -74.3 & dropoff_longitude <= -73)

cat("Dimention of Test data after removing outlier from longitude & latitude", dim(train))


# Scatter plot of Pickup Location
ggplot(aes(x = pickup_latitude, y = pickup_longitude), data = train)+
  geom_point(color = "blue")+
  ggtitle("Pickup Location")+
  xlab("Pickup Latitude")+
  ylab("Pickup Longitude")


# Scatter plot of Dropoff Location
ggplot(aes(x = dropoff_latitude, y = dropoff_longitude), data = train)+
  geom_point(color = "blue")+
  ggtitle("Dropoff Location")+
  xlab("Dropoff Latitude")+
  ylab("Dropoff Longitude")


######################################
#                                    #
#  Explore variable passenger_count  #
#                                    #
######################################


# check the class of 'passenger_count'
cat("Data type of passenger_count", class(train$passenger_count))

# convert passenger count from numeric to integer
train$passenger_count = as.integer(train$passenger_count)

# after conversion once again check the class of 'passenger_count'
cat("Data type of passenger_count", class(train$passenger_count))


summary(train$passenger_count)
summary(test$passenger_count)

cat("Below detail of Test Data")
cat("There are", nrow(filter(test, passenger_count == 0)), "observation where Passenger count is equal to 0.")
cat("There are", nrow(filter(test, passenger_count < 6 )), "observation where Passenger count is Less than 6.")
cat("There are", nrow(filter(test, passenger_count == 6)), "observation where Passenger count is equal to 6.")


cat("Below detail of Train Data")
cat("There are", nrow(filter(train, passenger_count == 0)), "observation where Passenger count is equal to 0.")
cat("There are", nrow(filter(train, passenger_count < 6 )), "observation where Passenger count is Less than 6.")
cat("There are", nrow(filter(train, passenger_count == 6)), "observation where Passenger count is equal to 6.")
cat("There are", nrow(filter(train, passenger_count > 6)), "observation where Passenger count is Greater than 6.")

# Bar plot of Passenger count with outliers
ggplot(aes(x = factor(passenger_count)), data = train)+
  geom_bar(fill = "blue")+
  geom_text(aes(label = percent(..count../sum(..count..))), stat = 'count', vjust = -0.5)+
  ggtitle("Bar Plot of Passenger Count with outliers")+
  xlab("Passenger Count")+
  ylab("Frequency")


cat("Dimension of Train data before removing outlier from passenger count", dim(train))


# remove outlier from passenger count
train = train %>%
  filter(passenger_count > 0 & passenger_count <= 6)

cat("Dimension of Train data after removing outlier from passenger count", dim(train))

# check correlation between passenger_count & fare amount
cor(train$passenger_count,train$fare_amount)


# Bar graph of passenger count without outliers
ggplot(aes(x = factor(passenger_count)), data = train)+
  geom_bar(fill = "blue")+
  scale_x_discrete(limits = c("0","1","2","3","4","5","6"))+
  geom_text(aes(label = percent(..count../sum(..count..))), stat = 'count', vjust = -0.5)+
  ggtitle("Bar Plot of Passenger Count without outliers")+
  xlab("Passenger Count")+
  ylab("Frequency")



######################################
#                                    #
#  Explore variable pickup_datetime  #
#                                    #
######################################

# before conversion check class
class(train$pickup_datetime)
class(test$pickup_datetime)


# convert pickup_datetime to POSIXct
train$pickup_datetime = as.POSIXct(train$pickup_datetime, format = "%Y-%m-%d %H:%M:%S")
test$pickup_datetime = as.POSIXct(test$pickup_datetime, format = "%Y-%m-%d %H:%M:%S")

# after conversion check class
class(train$pickup_datetime)
class(test$pickup_datetime)

# check missing value after converting pickup_datetime to POSIXct
apply(train,2,function(x){sum(is.na(x))})
apply(test,2,function(x){sum(is.na(x))})

cat("Dimention of Train Data before removing missing value", dim(train))

# remove missing value from Train data
train = na.omit(train)

cat("Dimention of Train Data after removing missing value", dim(train))

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
absolute = function(df){
  df %>%
    mutate(
      abs_lat_diff = abs(dropoff_latitude - pickup_latitude),
      abs_lon_diff = abs(dropoff_longitude - pickup_longitude)
    )
}

# create variables for absolut different between longitude & latitude
train = absolute(train)
test = absolute(test)


# check correlation between abs variables and fare amount
abs_var = grep("abs",colnames(train))

round(cor(train[,1], train[,abs_var]),2)


############################
#                          #
#    create new feature    #
#        'distance'        #
#                          #
############################


# define function to Compute Haversine distance
haversine_dis = function(df){
  df %>%
    mutate(
      distance = distHaversine(cbind(pickup_longitude, pickup_latitude),
                               cbind(dropoff_longitude, dropoff_latitude),
                               r = 6371)
    )
}

# compute Haversine distance for Train & Test data
train = haversine_dis(train)
test = haversine_dis(test)

# check correlation between distance & fare amount
round(cor(train$distance, train$fare_amount),2)

ggplot(aes(x = distance, y = fare_amount), data = train)+
  geom_point(color = "blue")+
  ggtitle("Scatter Plot of Distance & Fare amount")+
  xlab("Distance")+
  ylab("Fare Amount")

summary(train$distance)
summary(test$distance)



###########################
#                         #
#   create time series    #
#        features         #
#                         #
###########################

# define function create_date_time for variable 'pickup_datetime'
create_date_time = function(df){
  df %>%
    mutate(hour_of_day = as.numeric(format(pickup_datetime, "%H"))) %>%
    mutate(month = as.factor(format(pickup_datetime, "%m"))) %>%
    mutate(year = as.factor(format(pickup_datetime, "%Y"))) %>%
    mutate(day = as.numeric(format(pickup_datetime, "%d"))) %>%
    mutate(weekday = as.factor(weekdays(pickup_datetime)))
}

# create new variables in Train & Test data for Date & Time
train = create_date_time(train)
test = create_date_time(test)


# # histogram of hour_of_day to check frequency of rides during the day
ggplot(aes(x = factor(hour_of_day)), data = train)+
  geom_bar(fill = "blue")+
  geom_text(aes(label = percent(..count../sum(..count..))),stat = 'count', vjust = -0.5)+
  ggtitle("Bar plot of weekday count")+
  xlab("Hour of the Day")+
  ylab("Frequency")


# # # histogram of weekday to check frequency of rides during the week
ggplot(aes(x = factor(weekday)), data = train)+
  geom_bar(fill = "blue")+
  geom_text(aes(label = percent(..count../sum(..count..))),stat = 'count', vjust = -0.5)+
  ggtitle("Bar plot of weekday count")+
  xlab("Weekdays")+
  ylab("Frequency")


# convert date & time related variables into numeric in train dataset
train$month = as.numeric(train$month)
train$weekday = as.numeric(train$weekday)
train$year = as.numeric(train$year)


# convert date & time related variables into numeric in test dataset
test$month = as.numeric(test$month)
test$weekday = as.numeric(test$weekday)
test$year = as.numeric(test$year)



####################################
#                                  #
#   create new features base on    #
#        popular locations         #
#                                  #
####################################


# define function to compute distance from and to airports
create_airport_dis = function(df){
  
  jfk = c(-73.778889, 40.639722)
  ewr = c(-74.168611, 40.6925)
  lga = c(-73.872611, 40.77725)
  sol = c(-74.0445, 40.6892)
  nyc = c(-74.0063889, 40.7141667)
  
  df %>%
    mutate(
      jfk_dist = distHaversine(cbind(pickup_longitude, pickup_latitude), jfk, r = 6371) +
        distHaversine(cbind(dropoff_longitude, dropoff_latitude), jfk, r = 6371),
      ewr_dist = distHaversine(cbind(pickup_longitude, pickup_latitude), ewr, r = 6371) +
        distHaversine(cbind(dropoff_longitude, dropoff_latitude), ewr, r = 6371),
      lga_dist = distHaversine(cbind(pickup_longitude, pickup_latitude), lga, r = 6371) +
        distHaversine(cbind(dropoff_longitude, dropoff_latitude), lga, r = 6371),
      sol_dist = distHaversine(cbind(pickup_longitude, pickup_latitude), sol, r = 6371) +
        distHaversine(cbind(dropoff_longitude, dropoff_latitude), sol, r = 6371),
      nyc_dist = distHaversine(cbind(pickup_longitude, pickup_latitude), nyc, r = 6371) +
        distHaversine(cbind(dropoff_longitude, dropoff_latitude), nyc, r = 6371)
    )
}

# create airport related variables for train and test data
train = create_airport_dis(train)
test = create_airport_dis(test)

# check correlation between distance variable and fare amount

distance_var = grep("dist", colnames(train))

round(cor(train[,1], train[,distance_var]),2)


# remove 'pickup_datetime' variables from train & test dataset as save for later use

train_datetime = train$pickup_datetime
test_datetime = test$pickup_datetime

train = train %>%
  select(-pickup_datetime)

test = test %>%
  select(-pickup_datetime)


# check correlation with target variable

# make correlation plot
train %>% correlate() %>% focus(fare_amount) %>%
  ggplot(aes(x = rowname, y = fare_amount))+
  geom_bar(stat = "identity", fill = "blue")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))+
  ylab("Correlation with Fare Amount")+
  ggtitle("Correlation with Target Variable")


########################################
#                                      #
#   split data to Train & validation   #
#                                      #
########################################

# create sample size
sample_size = floor(0.20 * nrow(train))

set.seed(123)
index = sample(1:nrow(train), sample_size)

# split data into train1 & valid part
train1 = train[-index,]
valid = train[index,]


cat("Dimention of train1 data set:", dim(train1))
cat("Dimention of valid data set:", dim(valid))


#####################################
#                                   #
#   Evaluation matrix RMSE & MAPE   #
#                                   #
#####################################

rmse_metrics = function(predicted, actual){
  sqrt(mean((predicted - actual)**2))
}

mape_metrics = function(predicted, actual){
  mean(abs((predicted - actual)/ actual))
}


###################################
#                                 #
#   Multiple Linear Regression    #
#        with all variables       #
#                                 #
###################################

lr_mod = lm(fare_amount ~., data = train1)

# make prediction of Linear Model for Train & Validation data
lr_train_pred = predict(lr_mod, train1)
lr_valid_pred = predict(lr_mod, valid)

# compute RMSE & MAPE errors for both Training & Validation dataset

cat("Multiple Linear Regression model with all variables RMSE & MAPE")

lr_tr = round(rmse_metrics(lr_train_pred, train1$fare_amount),2)
lr_vr = round(rmse_metrics(lr_valid_pred, valid$fare_amount),2)

lr_tm = round(mape_metrics(lr_train_pred, train1$fare_amount),2)
lr_vm = round(mape_metrics(lr_valid_pred, valid$fare_amount),2)

cat("Training RMSE  :",lr_tr)
cat("Validation RMSE:",lr_vr)


cat("Training MAPE   :",lr_tm)
cat("Validation MAPE :",lr_vm)

# make prediction of Multiple Linear Regression Model 1 for Final Test Dataset
lr_test_pred = predict(lr_mod, test)

# Density plot of Multiple Linear Regression 1 model

lares::mplot_density(tag = valid$fare_amount,
                     score = lr_valid_pred,
                     subtitle = "Fare Regression Model",
                     model_name = "Multiple Linear Regression 1")


# variable selection base on low AIC by below method.
best = step(lr_mod)


###################################
#                                 #
#   Multiple Linear Regression    #
#         with lowest AIC         #
#                                 #
###################################

lr_mod1 = lm(fare_amount ~ pickup_longitude + pickup_latitude + dropoff_longitude + 
               month + year + distance + jfk_dist + ewr_dist + sol_dist + 
               nyc_dist + abs_lat_diff + abs_lon_diff, data = train1)


# make prediction of Linear Model for Train & Validation data
lr1 = predict(lr_mod1, train1)
lr2 = predict(lr_mod1, valid)

# compute RMSE & MAPE errors for both Training & Validation dataset

lr1_tr = round(rmse_metrics(lr1, train1$fare_amount),2)
lr1_vr = round(rmse_metrics(lr2, valid$fare_amount),2)

lr1_tm = round(mape_metrics(lr1, train1$fare_amount),2)
lr1_vm = round(mape_metrics(lr2, valid$fare_amount),2)

cat("Multiple Linear Regression model with selected variables RMSE & MAPE")

cat("Training RMSE  :", lr1_tr)
cat("Validation RMSE:", lr1_vr)

cat("Training MAPE   :", lr1_tr)
cat("Validation MAPE :", lr1_vr)

# make prediction of Multiple Linear Regression Model 2 for Final Test Dataset
lr1_test_pred = predict(lr_mod1, test)

# Density plot of Multiple Linear Regression 2 model

lares::mplot_density(tag = valid$fare_amount,
                     score = lr2,
                     subtitle = "Fare Regression Model",
                     model_name = "Multiple Linear Regression 2")


#rm(list = 'lr_mod')
#rm(list = 'lr_mod1')
#rm(list = 'best')


#####################
#                   #
#   Random Forest   #     
#                   #               
#####################

# use Random Fores model with all the variables & withour any hyper parameter tuning
rf = randomForest(fare_amount ~., data = train1)

# make prediction of Random Forest for Train & Validation data
rf_train_pred = predict(rf, train1)
rf_valid_pred = predict(rf, valid)


# compute RMSE & MAPE errors for both Training & Validation dataset

rf_tr = round(rmse_metrics(rf_train_pred, train1$fare_amount),2)
rf_vr = round(rmse_metrics(rf_valid_pred, valid$fare_amount),2)

rf_tm = round(mape_metrics(rf_train_pred, train1$fare_amount),2)
rf_vm = round(mape_metrics(rf_valid_pred, valid$fare_amount),2)

cat("Random Forest model with with default parameter setting")

cat("Training RMSE  :",rf_tr)
cat("Validation RMSE:",rf_vr)

cat("Training MAPE   :",rf_tm)
cat("Validation MAPE :",rf_vm)

# make prediction of Random Forest Default for Final Test Dataset
rf_test_pred = predict(rf, test)

# Density plot of Random Forest 1 model
lares::mplot_density(tag = valid$fare_amount,
                     score = rf_valid_pred,
                     subtitle = "Fare Regression Model",
                     model_name = "Random Forest 1")


# make dataframe for feature Importance as per Random Forest
feat_importances = data.frame(rf$importance)
feat_importances$feature = rownames(feat_importances)
names(feat_importances)[1] = "variable_imp"
rownames(feat_importances) = NULL
feat_importances = feat_importances[,c(2,1)]



# Bar plot of Feature Importance.
ggplot(aes(x = feature, y = variable_imp), data = feat_importances)+
  geom_bar(stat = "identity", fill = "blue")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))+
  ylab("Relative Feature Importance")+
  ggtitle("Feature Importance")




#######################################
#                                     #
#        Random Forest with           #
# cross-Validation & with 1000 trees  #
#                                     #
#######################################

control = trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 3,
                       search = "grid")



# use Random Forest with hyper parameter tuning
rf_mod1 = randomForest(fare_amount ~.,
                       data = train1,
                       method = "rf",
                       metric = 'rmse',
                       ntree = 1000,
                       trControl = control)


# make prediction of Random Forest for Train & Validation data
rf1 = predict(rf_mod1, train1)
rf2 = predict(rf_mod1, valid)


# compute RMSE & MAPE errors for both Training & Validation dataset

rf1_tr = round(rmse_metrics(rf1, train1$fare_amount),2)
rf1_vr = round(rmse_metrics(rf2, valid$fare_amount),2)

rf1_tm = round(mape_metrics(rf1, train1$fare_amount),2)
rf1_vm = round(mape_metrics(rf2, valid$fare_amount),2)

cat("Random Forest model with 1000 number of trees & grid search")

cat("Training RMSE  :",rf1_tr)
cat("Validation RMSE:",rf1_vr)

cat("Training MAPE   :",rf1_tm)
cat("Validation MAPE :",rf1_vm)

rf1_test_pred = predict(rf_mod1, test)

# Density plot of Random Forest 2 model

lares::mplot_density(tag = valid$fare_amount,
                     score = rf2,
                     subtitle = "Fare Regression Model",
                     model_name = "Random Forest 2")


#rm(list = "rf")
#rm(list = 'rf_mod1')



###############
#             #      
#   XGBoost   #     
#             #                    
###############

# convert dataframe in matrix
dtrain1 = xgb.DMatrix(data = data.matrix(train1[,-1]), label = train1[,1])
dvalid = xgb.DMatrix(data = data.matrix(valid[,-1]), label = valid[,1])

xgb_parameter = list(objective = "reg:linear",
                     eval_metric = "rmse",
                     num_boost_round = 300,
                     nrounds = 300)

xgb = xgb.train(xgb_parameter,dtrain1, xgb_parameter$nrounds, list(val = dvalid),
                print_every_n = 20,
                early_stopping_rounds = 10)


# make prediction of XGBoost for Train & Validation data
xgb_train_pred = predict(xgb, dtrain1)
xgb_valid_pred = predict(xgb, dvalid)


# compute RMSE & MAPE errors for both Training & Validation dataset

xgb_tr = round(rmse_metrics(xgb_train_pred, train1$fare_amount),2)
xgb_vr = round(rmse_metrics(xgb_valid_pred, valid$fare_amount),2)

xgb_tm = round(mape_metrics(xgb_train_pred, train1$fare_amount),2)
xgb_vm = round( mape_metrics(xgb_valid_pred, valid$fare_amount),2)

cat("XGBoost model")

cat("Training RMSE  :",xgb_tr)
cat("Validation RMSE:",xgb_vr)

cat("Training MAPE   :",xgb_tm)
cat("Validation MAPE :",xgb_vm)

# make prediction of XGBoost for Final Test Dataset
dtest = xgb.DMatrix(data = data.matrix(test))
xgb_test_pred = predict(xgb, dtest)

# Density plot of XGBoost model

lares::mplot_density(tag = valid$fare_amount,
                     score = xgb_valid_pred,
                     subtitle = "Fare Regression Model",
                     model_name = "XGBoost")




###################################
#                                 #
#    Compare the performance of   #
#         all used models         #
#                                 #
###################################

# make data frame 'all_model' for store RMSE score of all used model

all_model = data.frame("model" = c("Multiple Linear Regression 1",
                                   "Multiple Linear Regression 2",
                                   "Random Forest 1", "Random Forest 2",
                                   "XGBoost"),
                       "train_rmse" = c(lr_tr, lr1_tr, rf_tr, rf1_tr, xgb_tr),
                       "valid_rmse" = c(lr_vr, lr1_vr, rf_vr, rf1_vr, xgb_vr))


# Bar graph of model performance
ggplot(data = all_model, aes(x = model, y = valid_rmse))+
  geom_bar(stat = "identity", fill = "blue")+
  coord_flip()+
  ggtitle("Compare Performance of all used Models")+
  xlab("Models")+
  ylab("Root Mean Square Error")




#################################
#                               #
#   Submission result of all    #
#         used models           #
#                               #
#################################

# add Pickup_datetime which was removed while modeling
test$pickup_datetime = test_datetime

# sumission of Linear Regression 1
test$fare_amount_linear_1 = lr_test_pred

# submission of Linear Regression 2
test$fare_amount_linear_2 = lr1_test_pred

# Submission of Random Forest 1
test$fare_amount_random_for_1 = rf_test_pred

# submission of Random Forest 2
test$fare_amount_random_for_2 = rf1_test_pred

# submission of XGBoost
test$fare_amount_xgboost = xgb_test_pred


# write file to working directory
write.csv(test, file = "R_sub_all_models.csv", row.names = FALSE)
