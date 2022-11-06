# Libraries
library(glmnet)


# Source: https://www.kaggle.com/code/sarvaninandipati/analysis-prediction-of-walmart-sales-using-r/notebook

# Get Datasets
#train <- readr::read_csv('train_ini.csv')
#test <- readr::read_csv('test.csv')


mypredict <- function() {
  
  # #Preprocessing train
  # data4 <- train
  # 
  # data4['IsHoliday'] = as.integer(as.logical(data4$IsHoliday))
  # 
  # #formatting date to dd-mm-yyyy
  # data4$Date <- format(data4$Date, "%d-%m-%Y")
  # 
  # #changing date column in dataframe to date format & arranging in ascending order as per dates
  # data4$Date <- lubridate::dmy(data4$Date)
  # data4 <- dplyr::arrange(data4,Date)
  # 
  # #Creating a week number,month,quarter column in dataframe
  # data4$Week_Number <- lubridate::week(data4$Date)
  # 
  # #adding quarter & month columns
  # data4$month <- lubridate::month(data4$Date)
  # data4$quarter <- lubridate::quarter(data4$Date)
  # data4$year <- lubridate::year(data4$Date)
  # 
  # 
  # ##Creating a event type dataframe##
  # 
  # # creating Holiday_date vector
  # Holiday_date <- c("12-02-2010", "11-02-2011", "10-02-2012", "08-02-2013","10-09-2010", "09-09-2011", "07-09-2012", "06-09-2013","26-11-2010", "25-11-2011", "23-11-2012", "29-11-2013","31-12-2010", "30-12-2011", "28-12-2012", "27-12-2013")
  # 
  # #assigning date format to Holiday_date vector
  # Holiday_date <- lubridate::dmy(Holiday_date)
  # 
  # #Creating Events vector
  # Events <-c(rep("Super_Bowl", 4), rep("Labor_Day", 4),rep("Thanksgiving", 4), rep("Christmas", 4))
  # 
  # #Creating dataframe with Events and date
  # Holidays_Data <- data.frame(Events,Holiday_date)
  # 
  # #merging both dataframes
  # data4<-merge(data4,Holidays_Data, by.x= "Date", by.y="Holiday_date", all.x = TRUE)
  # 
  # #Replacing null values in Event with No_Holiday
  # data4$Events = as.character(data4$Events)
  # data4$Events[is.na(data4$Events)]= "No_Holiday"
  # 
  # #Convert to factor and numeric
  # categorical.vars = c("Super_Bowl", "Labor_Day", "Thanksgiving", "Christmas", "No_Holiday")
  # 
  # for(var in categorical.vars) {
  #   data4[var] = data4$Events == var
  # }
  # 
  # cols <- sapply(data4, is.logical)
  # data4[,cols] <- lapply(data4[,cols], as.numeric)
  # 
  # data4[data4$year == 2010, "Week_Number"] = data4[data4$year == 2010,"Week_Number"] - 1
  # 
  # 
  # #Drop Unwanted columns and create train.x and train.y
  # x_train_drop <- c('Date', 'Weekly_Sales', 'Events')
  # train.x = data4[,!(names(data4) %in% x_train_drop)]
  # train.y = train['Weekly_Sales']
  # 
  # 
  # 
  # 
  # #Preprocessing test
  # data5 <- test
  # 
  # data5['IsHoliday'] = as.integer(as.logical(data5$IsHoliday))
  # 
  # #formatting date to dd-mm-yyyy
  # data5$Date <- format(data5$Date, "%d-%m-%Y")
  # 
  # #changing date column in dataframe to date format & arranging in ascending order as per dates
  # data5$Date <- lubridate::dmy(data5$Date)
  # data5 <- dplyr::arrange(data5,Date)
  # 
  # #Creating a week number,month,quarter column in dataframe
  # data5$Week_Number <- lubridate::week(data5$Date)
  # 
  # #adding quarter & month columns
  # data5$month <- lubridate::month(data5$Date)
  # data5$quarter <- lubridate::quarter(data5$Date)
  # data5$year <- lubridate::year(data5$Date)
  # 
  # 
  # ##Creating a event type dataframe##
  # 
  # # creating Holiday_date vector
  # Holiday_date <- c("12-02-2010", "11-02-2011", "10-02-2012", "08-02-2013","10-09-2010", "09-09-2011", "07-09-2012", "06-09-2013","26-11-2010", "25-11-2011", "23-11-2012", "29-11-2013","31-12-2010", "30-12-2011", "28-12-2012", "27-12-2013")
  # 
  # #assigning date format to Holiday_date vector
  # Holiday_date <- lubridate::dmy(Holiday_date)
  # 
  # #Creating Events vector
  # Events <-c(rep("Super_Bowl", 4), rep("Labor_Day", 4),rep("Thanksgiving", 4), rep("Christmas", 4))
  # 
  # #Creating dataframe with Events and date
  # Holidays_Data <- data.frame(Events,Holiday_date)
  # 
  # #merging both dataframes
  # data5<-merge(data5,Holidays_Data, by.x= "Date", by.y="Holiday_date", all.x = TRUE)
  # 
  # #Replacing null values in Event with No_Holiday
  # data5$Events = as.character(data5$Events)
  # data5$Events[is.na(data5$Events)]= "No_Holiday"
  # 
  # #Convert to factor and numeric
  # categorical.vars = c("Super_Bowl", "Labor_Day", "Thanksgiving", "Christmas", "No_Holiday")
  # 
  # for(var in categorical.vars) {
  #   data5[var] = data5$Events == var
  # }
  # 
  # cols <- sapply(data5, is.logical)
  # data5[,cols] <- lapply(data5[,cols], as.numeric)
  # 
  # data5[data5$year == 2010, "Week_Number"] = data5[data5$year == 2010,"Week_Number"] - 1
  # 
  # 
  # #Drop Unwanted columns and create train.x and train.y
  # x_train_drop <- c('Date', 'Weekly_Sales', 'Events')
  # test.x = data5[,!(names(data5) %in% x_train_drop)]
  # 
  # 
  # #model = lm(formula = as.matrix(train.y) ~ . , data = train.x)
  # #y_pred_train = predict(model, newdata = test.x)
  # #y_pred_train
  # 
  # #Lasso Regression
  # cv.out = cv.glmnet(as.matrix(train.x), as.matrix(train.y), alpha = 0)
  # best.lam = cv.out$lambda.min
  # #best.lam = cv.out$lambda.1se
  # Ytest.pred = predict(cv.out, s = best.lam, newx = as.matrix(test.x))
  # Ytest.pred.df = data.frame(Date = test[3], Store = test[1], Dept = test[2], Weekly_Pred = Ytest.pred)
  # colnames(Ytest.pred.df)[4]<-"Weekly_Pred"
  # 
  # 
  # 
  # # date weekly_pred
  # 
  # return(Ytest.pred.df)
  
  train['IsHoliday'] = as.numeric(unlist(train['IsHoliday']))*4+1
  test['IsHoliday'] = as.numeric(unlist(test['IsHoliday']))*4+1
  
  train_pairs <- train[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])
  
  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    left_join(train, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(lubridate::year(Date) == 2010, lubridate::week(Date) - 1, lubridate::week(Date)), levels = 1:52)) %>% 
    mutate(Yr = lubridate::year(Date))
  train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Yr + Wk, train_split)) %>% group_split(Store, Dept)
  
  # do the same for the test set
  test_split <- unique_pairs %>% 
    left_join(test, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(lubridate::year(Date) == 2010, lubridate::week(Date) - 1, lubridate::week(Date)), levels = 1:52)) %>% 
    mutate(Yr = lubridate::year(Date))
  test_split = as_tibble(model.matrix(~ Store + Dept + Yr + Wk, test_split)) %>% mutate(Date = test_split$Date) %>% group_split(Store, Dept)
  
  # pre-allocate a list to store the predictions
  test_pred <- vector(mode = "list", length = nrow(unique_pairs))
  
  # perform regression for each split, note we used lm.fit instead of lm
  for (i in 1:nrow(unique_pairs)) {
    #tmp_train <- as.data.frame(train_split[[i]])
    #tmp_test <- as.data.frame(test_split[[i]])
    tmp_train <- as.matrix(train_split[[i]])
    tmp_test <- as.matrix(test_split[[i]])
    
    tmp_train = tmp_train[, -1]
    tmp_test = tmp_test[, -1]
    
    X_m_n = 
    # construct matrix M stores, n weeks
    # Ith row and jth column corresponds to the weekly sales  at Is store and Js week
    # use svd function in r to get bullet point 2 (could also try prcomp) to get UDV matrix
    
    #pca_tmp_train <- prcomp(tmp_train, center = FALSE, scale = FALSE)
    
    #F1_train = tmp_train%*%pca_tmp_train$rotation
    #F1_test = Xtest%*%pca_tmp_train$rotation
    
    # mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
    # mycoef[is.na(mycoef)] <- 0
    # tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:55]) %*% mycoef[-1]
    fit <- lm(formula = tmp_train$Weekly_Sales ~ . + I(Yr ^2) + I(Yr ^3), data = tmp_train[, -(1:3)])
    #mycoef <- fit$coefficients
    #tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:55]) %*% mycoef[-(1:2)]
    tmp_pred = predict(fit, tmp_test[, 3:54])
    
    test_pred[[i]] <- cbind(tmp_test[, 1:2], Date = tmp_test$Date, Weekly_Pred = tmp_pred)
  }
  
  # turn the list into a table at once, 
  # this is much more efficient then keep concatenating small tables
  test_pred <- bind_rows(test_pred)
  
  return(test_pred)
}


