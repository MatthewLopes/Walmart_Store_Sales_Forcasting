# project_2
Project 2: Walmart Store Sales Forcasting

Fall 2022

You are provided with historical sales data for 45 Walmart stores located in different regions. Each store contains many departments. The goal is to predict the future weekly sales for each department in each store based on the historical data.

Goal

The file, train_ini.csv, provides the weekly sales data for various stores and departments from 2010-02 (February 2010) to 2011-02 (February 2011).

Given train_ini.csv, the data till 2011-02, you need to predict the weekly sales for 2011-03 and 2011-04. Then you’ll be provided with the weekly sales data for 2011-03 and 2011-04 (fold_1.csv), and you need to predict the weekly sales for 2011-05 and 2011-06, and so on:

t = 1, predict 2011-03 to 2011-04 based on data from 2010-02 to 2011-02 (train_ini.csv);
t = 2, predict 2011-05 to 2011-06 based on data from 2010-02 to 2011-04 (train_ini.csv, fold_1.csv);
t = 3, predict 2011-07 to 2011-08 based on data from 2010-02 to 2011-06 (train_ini.csv, fold_1.csv, fold_2.csv);
……
t = 10, predict 2012-09 to 2012-10 based on data from 2010-02 to 2012-08 (train_ini.csv, fold_1.csv, fold_2.csv, …, fold_9.csv)
