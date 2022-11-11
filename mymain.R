# Libraries
library(glmnet)
library(dplyr)

# Source: https://www.kaggle.com/code/sarvaninandipati/analysis-prediction-of-walmart-sales-using-r/notebook

# Get Datasets
#train <- readr::read_csv('train_ini.csv')
#test <- readr::read_csv('test.csv')


mypredict <- function() {
  
  train_df = as.data.frame(train)
  
  ## Move into X matrix
  
  for(dpt_num in unique(train$Dept)) {
    
    #filter data by dept
    dept_data = train %>% filter(Dept == dpt_num)
    
    #initialize mxn m is # stores with dept, n # of weeks
    m = max(unique(dept_data$Store))
    n = length(unique(dept_data$Date))
    X_m_n = matrix(0, m, n)
    
    unique_stores = unique(dept_data$Store)
    
    for(store_num in unique_stores){
      q = dept_data %>% filter(Store == store_num)
      date_indicies= c(1:dim(q)[1])
      X_m_n[store_num,date_indicies] = q$Weekly_Sales
      
    }
    X_m_n_list[[dpt_num]] = X_m_n
  }
  
  ## Apply SVD and bring back into Train
  
  for(dpt_num in unique(train_df$Dept)) {
    U_D_V = svd(X_m_n_list[[dpt_num]])
    
    r = min(dim(U_D_V$u)[2], 8)
    
    d_new = diag(U_D_V$d[1:r], r, r)
    u_new = U_D_V$u[,1:r]
    v_new = U_D_V$v[,1:r]
    
    x_tilda = u_new %*% d_new %*% t(v_new)
    
    dept_data = train %>% filter(Dept == dpt_num)
    
    unique_stores = unique(dept_data$Store)
    
    for(store_num in unique_stores){
      p = dept_data %>% filter(Store == store_num)
      date_indicies= c(1:dim(p)[1])
      p$Weekly_Sales = x_tilda[store_num,date_indicies]
      
      train_df[train_df$Dept == dpt_num & train_df$Store == store_num, ]$Weekly_Sales = x_tilda[store_num,date_indicies]
      
    }
    
  }
  
  train_pairs <- train_df[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])
  
  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    left_join(train_df, by = c('Store', 'Dept')) %>% 
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
    tmp_train <- as.data.frame(train_split[[i]])
    tmp_test <- as.data.frame(test_split[[i]])
    # tmp_train <- as.matrix(train_split[[i]])
    # tmp_test <- as.matrix(test_split[[i]])
    
    tmp_train = tmp_train[, -1]
    tmp_test = tmp_test[, -1]
    
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


