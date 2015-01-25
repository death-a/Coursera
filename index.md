<u>Practical Machine Learning - Course Project</u>
========================================================

The Main Goal of this Project is to Predict to which <i>"Classe"</i> the performed exercise belongs to. We are given a training data set & a test data set seperately. The raw training data set contains untidy data which will not at help in fufilling our goal. So it is necessary to clean up the training data set in order for the prediction to work.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.1.2
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.1.2
```

```r
set.seed(2804)
trainingfit <- read.csv("~/pml-training.csv")
testingfit <- read.csv("~/pml-testing.csv")
```

## <b>Data Cleaning</b>
Once the training & testing data is loaded, we can get the summary and structure information of the training data set using the following functions:


```r
## Displaying only select summary for demo purpose
summary(trainingfit[1:10])
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3892   Min.   :1.322e+09    Min.   :   294      
##  1st Qu.: 4906   carlitos:3112   1st Qu.:1.323e+09    1st Qu.:252912      
##  Median : 9812   charles :3536   Median :1.323e+09    Median :496380      
##  Mean   : 9812   eurico  :3070   Mean   :1.323e+09    Mean   :500656      
##  3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.323e+09    3rd Qu.:751891      
##  Max.   :19622   pedro   :2610   Max.   :1.323e+09    Max.   :998801      
##                                                                           
##           cvtd_timestamp  new_window    num_window      roll_belt     
##  28/11/2011 14:14: 1498   no :19216   Min.   :  1.0   Min.   :-28.90  
##  05/12/2011 11:24: 1497   yes:  406   1st Qu.:222.0   1st Qu.:  1.10  
##  30/11/2011 17:11: 1440               Median :424.0   Median :113.00  
##  05/12/2011 11:25: 1425               Mean   :430.6   Mean   : 64.41  
##  02/12/2011 14:57: 1380               3rd Qu.:644.0   3rd Qu.:123.00  
##  02/12/2011 13:34: 1375               Max.   :864.0   Max.   :162.00  
##  (Other)         :11007                                               
##    pitch_belt          yaw_belt      
##  Min.   :-55.8000   Min.   :-180.00  
##  1st Qu.:  1.7600   1st Qu.: -88.30  
##  Median :  5.2800   Median : -13.00  
##  Mean   :  0.3053   Mean   : -11.21  
##  3rd Qu.: 14.9000   3rd Qu.:  12.90  
##  Max.   : 60.3000   Max.   : 179.00  
## 
```

```r
## Displaying only select variable structure for demo purpose
str(trainingfit,list.len=10)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##   [list output truncated]
```

From the summary and str function on the training data set, the useful <b>features can be extracted</b> and the ones with <b>NA</b> values or with a <b>divide by zero</b> error, in this case, can be ignored. 
Therefore, I extracted the features that contained proper values which could be used for creating a model in order to predict on the test data set. The proper predictors were:
* The Euler angles (roll, pitch & yaw) for all the four sensors, 
* The accelerometers, gyroscope & magnetometer readings for all the four sensors on all the three axis and 
* The Total accelerometers reading specifically for all four sensors.

Thus, we can get a <b>clean training data set</b> with the help of following commands:

```r
myvars <- names(trainingfit) %in% c("user_name","roll_belt","pitch_belt","yaw_belt","total_accel_belt","gyros_belt_x"
,"gyros_belt_y","gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z"
,"magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm","yaw_arm"
,"total_accel_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x"
,"accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z"
,"roll_dumbbell","pitch_dumbbell","yaw_dumbbell","total_accel_dumbbell"
,"gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x"
,"accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y"
,"magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm"
,"total_accel_forearm","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z"
,"accel_forearm_x","accel_forearm_y","accel_forearm_z","magnet_forearm_x"
,"magnet_forearm_y","magnet_forearm_z","classe")

cleanFitData = trainingfit[myvars]
```

## <b>Model Building</b>
### Rpart
Using Rpart we can get a brief idea about the prediction that we can apply on the test data set inorder to predict the outcome i.e. classe for the test data. Refer following plot.

```r
fitRpart <- train(classe ~., data=cleanFitData[,-55], method="rpart")
fancyRpartPlot(fitRpart$finalModel)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-1.png) 
### Random Forest 
I built my model using <u>Random Forest</u> on the outcome variable <i>classe</i> and rest of the features, from the tidy train data set, as predictors.
It creates 200 trees as samples and generates a model taking the predictors importance into consideration.
The model is then applied to the test data set to predict <i>classe</i> on the test data set using the predict function to get the desired result to fulfill the goal.


```r
modfit <- randomForest(classe ~ ., data=cleanFitData[,-55], importance=TRUE, ntree=200)
modfit$confusion
```

```
##      A    B    C    D    E  class.error
## A 5579    1    0    0    0 0.0001792115
## B   11 3781    5    0    0 0.0042138530
## C    0   12 3408    2    0 0.0040911748
## D    0    0   24 3190    2 0.0080845771
## E    0    0    2    5 3600 0.0019406709
```

```r
predRF = predict(modfit,newdata=testingfit)
submit <- data.frame(classe = predRF)
submit
```

```
##    classe
## 1       B
## 2       A
## 3       B
## 4       A
## 5       A
## 6       E
## 7       D
## 8       B
## 9       A
## 10      A
## 11      B
## 12      C
## 13      B
## 14      A
## 15      E
## 16      E
## 17      A
## 18      B
## 19      B
## 20      B
```




