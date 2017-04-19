# ML_Project.Rmd


## Practical Machine Learning 
## Date : 4/9/2017 


### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

### Data:
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

### Goal:
The goal of the project is to predict the manner in which the exercise was done. This is the "classe" variable in the training set. Other variables may be used to predict with. A report describing how the model was built including how cross validation was used, the expected out of sample error and why the choices made were done. Also utilize the prediction model to predict 20 different test cases. 

### Loading data:

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.3.2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.3.3
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.3.3
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.3.3
```

```r
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
dim(training) ; dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```
## Data Cleaning: 
Let's do some cleaning before we split the data.
Let's delete rows with missing values and columns that may be contextual and not provide any prediction

```r
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
dim(training); dim(testing)
```

```
## [1] 19622    53
```

```
## [1] 20 53
```

Now partition the dataset into 2 pieces for training and validating

```r
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
myTraining <- training[inTrain, ]
myvalidating <- training[-inTrain, ]
dim(myTraining); dim(myvalidating)
```

```
## [1] 13737    53
```

```
## [1] 5885   53
```
### Prediction Algorithms:
We will use K-fold cross validation and Random forests to predict the outcome.

K-fold cross validation: 
Let's use the default 10 fold cross validation here. 


```r
foldcontrol <- trainControl(method = "cv", number = 10)
fitrpart <- train(classe ~ ., data = myTraining, method = "rpart", 
                   trControl = foldcontrol)
print(fitrpart, digits = 4)
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12361, 12364, 12364, 12365, 12364, 12362, ... 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa  
##   0.03672  0.5178    0.37450
##   0.06100  0.4425    0.25330
##   0.11596  0.3235    0.06021
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03672.
```


```r
fancyRpartPlot(fitrpart$finalModel)
```

![](ML-_Project_files/figure-html/unnamed-chunk-5-1.png)<!-- -->
## Predict outcomes using the "myvalidating " set

```r
predict_rpart <- predict(fitrpart, myvalidating)

## Show output
confusionMatrix(myvalidating$classe, predict_rpart)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1501   35  113    0   25
##          B  475  375  289    0    0
##          C  487   34  505    0    0
##          D  429  173  362    0    0
##          E  162  143  267    0  510
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4912          
##                  95% CI : (0.4784, 0.5041)
##     No Information Rate : 0.5189          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3351          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4915  0.49342  0.32878       NA  0.95327
## Specificity            0.9389  0.85093  0.88020   0.8362  0.89308
## Pos Pred Value         0.8967  0.32924  0.49220       NA  0.47135
## Neg Pred Value         0.6312  0.91888  0.78782       NA  0.99479
## Prevalence             0.5189  0.12914  0.26100   0.0000  0.09091
## Detection Rate         0.2551  0.06372  0.08581   0.0000  0.08666
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
## Balanced Accuracy      0.7152  0.67217  0.60449       NA  0.92318
```

```r
confusionMatrix(myvalidating$classe, predict_rpart)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.4912489      0.3351156      0.4783984      0.5041081      0.5189465 
## AccuracyPValue  McnemarPValue 
##      0.9999900            NaN
```

```r
confusionMatrix(myvalidating$classe, predict_rpart)$overall[1]
```

```
##  Accuracy 
## 0.4912489
```
The confustion Matric shows an accuracy rate with K-fold cross validation (10 fold) is only 0.49.

Now let's see what Neural Networks can do

```r
fit_nn <- train(classe ~ ., data = myTraining, method = "nnet", 
                   trControl = foldcontrol,  returnResamp = "all")
```

```
## Loading required package: nnet
```

```
## # weights:  63
## initial  value 21404.458770 
## iter  10 value 18536.250804
## iter  20 value 18431.494363
## iter  30 value 18362.446894
## iter  40 value 18025.785766
## iter  50 value 17953.336719
## iter  60 value 17939.233556
## iter  70 value 17929.321846
## iter  80 value 17878.683990
## iter  90 value 17862.099175
## iter 100 value 17828.473475
## final  value 17828.473475 
## stopped after 100 iterations
## # weights:  179
## initial  value 23797.674897 
## iter  10 value 18392.282110
## iter  20 value 17907.516235
## iter  30 value 17546.260219
## iter  40 value 17342.790789
## iter  50 value 17062.388476
## iter  60 value 16814.587236
## iter  70 value 16780.979497
## iter  80 value 16737.996693
## iter  90 value 16658.209927
## iter 100 value 16643.770449
## final  value 16643.770449 
## stopped after 100 iterations
## # weights:  295
## initial  value 21766.436415 
## iter  10 value 18737.043118
## iter  20 value 18277.708061
## iter  30 value 17861.700018
## iter  40 value 17562.742242
## iter  50 value 17092.667624
## iter  60 value 16708.661200
## iter  70 value 16535.667947
## iter  80 value 16479.806686
## iter  90 value 16350.659748
## iter 100 value 16114.011406
## final  value 16114.011406 
## stopped after 100 iterations
## # weights:  63
## initial  value 23077.043325 
## iter  10 value 19579.614915
## iter  20 value 19499.888034
## iter  30 value 19479.375969
## iter  40 value 19457.175605
## iter  50 value 19403.151410
## iter  60 value 19374.263797
## iter  70 value 19338.647919
## iter  80 value 19313.485686
## iter  90 value 19282.997859
## iter 100 value 19268.771337
## final  value 19268.771337 
## stopped after 100 iterations
## # weights:  179
## initial  value 22047.930522 
## iter  10 value 19342.280618
## iter  20 value 19143.582252
## iter  30 value 18928.659831
## iter  40 value 18439.385258
## iter  50 value 18411.867367
## iter  60 value 18189.074998
## iter  70 value 17918.790722
## iter  80 value 17832.406976
## iter  90 value 17605.889916
## iter 100 value 17536.306707
## final  value 17536.306707 
## stopped after 100 iterations
## # weights:  295
## initial  value 24011.265979 
## iter  10 value 19136.679554
## iter  20 value 18557.769486
## iter  30 value 18240.030625
## iter  40 value 18049.290088
## iter  50 value 17808.055015
## iter  60 value 17636.364626
## iter  70 value 17464.455062
## iter  80 value 17325.820625
## iter  90 value 17233.143666
## iter 100 value 17114.556690
## final  value 17114.556690 
## stopped after 100 iterations
## # weights:  63
## initial  value 19899.122195 
## iter  10 value 18526.042005
## iter  20 value 18002.989420
## iter  30 value 17669.129695
## iter  40 value 17533.786934
## iter  50 value 17391.899389
## iter  60 value 17276.622650
## iter  70 value 17202.314031
## iter  80 value 17178.046929
## iter  90 value 17122.821646
## iter 100 value 17107.310700
## final  value 17107.310700 
## stopped after 100 iterations
## # weights:  179
## initial  value 19993.753973 
## iter  10 value 18320.293610
## iter  20 value 17960.408154
## iter  30 value 17893.146227
## iter  40 value 17787.943139
## iter  50 value 17644.969947
## iter  60 value 17323.751536
## iter  70 value 17120.395984
## iter  80 value 16986.005690
## iter  90 value 16928.335587
## iter 100 value 16884.788798
## final  value 16884.788798 
## stopped after 100 iterations
## # weights:  295
## initial  value 21379.899767 
## iter  10 value 19596.505726
## iter  20 value 19185.947342
## iter  30 value 18488.639094
## iter  40 value 18025.327824
## iter  50 value 17793.636187
## iter  60 value 17705.633833
## iter  70 value 17507.345819
## iter  80 value 17429.054141
## iter  90 value 17378.403306
## iter 100 value 17277.716422
## final  value 17277.716422 
## stopped after 100 iterations
## # weights:  63
## initial  value 21244.341616 
## iter  10 value 19299.069594
## iter  20 value 19262.852717
## iter  30 value 19100.233010
## iter  40 value 18784.695842
## iter  50 value 18775.000492
## iter  60 value 18770.458173
## iter  70 value 18702.712090
## iter  80 value 18439.614208
## iter  90 value 18365.772548
## iter 100 value 18148.457910
## final  value 18148.457910 
## stopped after 100 iterations
## # weights:  179
## initial  value 22756.287949 
## iter  10 value 18734.991544
## iter  20 value 17992.515053
## iter  30 value 17706.794459
## iter  40 value 17493.037004
## iter  50 value 17379.317143
## iter  60 value 17240.658613
## iter  70 value 17146.289190
## iter  80 value 17048.540068
## iter  90 value 16966.740550
## iter 100 value 16905.093431
## final  value 16905.093431 
## stopped after 100 iterations
## # weights:  295
## initial  value 23353.936323 
## iter  10 value 18455.801895
## iter  20 value 18015.660436
## iter  30 value 17758.143126
## iter  40 value 17584.316323
## iter  50 value 17535.667234
## iter  60 value 17484.633778
## iter  70 value 17444.838556
## iter  80 value 17409.624733
## iter  90 value 17383.738226
## iter 100 value 17347.877912
## final  value 17347.877912 
## stopped after 100 iterations
## # weights:  63
## initial  value 22089.205020 
## iter  10 value 18611.570412
## iter  20 value 18401.814096
## iter  30 value 18349.463352
## iter  40 value 18298.866051
## iter  50 value 18204.790250
## iter  60 value 18137.249605
## iter  70 value 18095.542040
## iter  80 value 18008.930019
## iter  90 value 17937.607158
## iter 100 value 17712.325335
## final  value 17712.325335 
## stopped after 100 iterations
## # weights:  179
## initial  value 20841.144098 
## iter  10 value 19148.346834
## iter  20 value 18557.682639
## iter  30 value 18211.203544
## iter  40 value 17956.020401
## iter  50 value 17861.753001
## iter  60 value 17791.701071
## iter  70 value 17694.311933
## iter  80 value 17620.347807
## iter  90 value 17475.442727
## iter 100 value 17389.536837
## final  value 17389.536837 
## stopped after 100 iterations
## # weights:  295
## initial  value 24490.899923 
## iter  10 value 18997.304832
## iter  20 value 18775.918725
## iter  30 value 17763.503278
## iter  40 value 17382.096462
## iter  50 value 17238.718145
## iter  60 value 17074.894046
## iter  70 value 16682.186241
## iter  80 value 16501.343406
## iter  90 value 16278.549129
## iter 100 value 16190.346942
## final  value 16190.346942 
## stopped after 100 iterations
## # weights:  63
## initial  value 20182.065224 
## iter  10 value 19116.210207
## iter  20 value 19063.705991
## iter  30 value 18993.406050
## iter  40 value 18675.137523
## iter  50 value 18608.961160
## iter  60 value 18459.231409
## iter  70 value 18429.739433
## iter  80 value 18354.757951
## iter  90 value 18348.863551
## iter 100 value 18343.197057
## final  value 18343.197057 
## stopped after 100 iterations
## # weights:  179
## initial  value 20827.451175 
## iter  10 value 19127.180369
## iter  20 value 18949.159034
## iter  30 value 18872.378681
## iter  40 value 18701.011081
## iter  50 value 18333.238845
## iter  60 value 18261.935470
## iter  70 value 17840.519229
## iter  80 value 17766.194004
## iter  90 value 17667.821950
## iter 100 value 17617.558511
## final  value 17617.558511 
## stopped after 100 iterations
## # weights:  295
## initial  value 20636.221341 
## iter  10 value 18939.643542
## iter  20 value 18649.274090
## iter  30 value 18195.226286
## iter  40 value 17882.548103
## iter  50 value 17601.087141
## iter  60 value 17473.467675
## iter  70 value 17326.460202
## iter  80 value 17295.094115
## iter  90 value 17252.812801
## iter 100 value 17196.622310
## final  value 17196.622310 
## stopped after 100 iterations
## # weights:  63
## initial  value 20778.043405 
## iter  10 value 18872.249452
## iter  20 value 18820.499444
## iter  30 value 18156.084150
## iter  40 value 17982.392369
## iter  50 value 17851.166370
## iter  60 value 17801.896039
## iter  70 value 17689.712150
## iter  80 value 17646.430692
## iter  90 value 17623.869420
## iter 100 value 17584.344352
## final  value 17584.344352 
## stopped after 100 iterations
## # weights:  179
## initial  value 21547.832704 
## iter  10 value 19135.025492
## iter  20 value 18390.876322
## iter  30 value 18020.922737
## iter  40 value 17968.186414
## iter  50 value 17916.289882
## iter  60 value 17898.187980
## iter  70 value 17859.717713
## iter  80 value 17755.947164
## iter  90 value 17687.927930
## iter 100 value 17605.519612
## final  value 17605.519612 
## stopped after 100 iterations
## # weights:  295
## initial  value 20688.067998 
## iter  10 value 18340.617479
## iter  20 value 17779.240981
## iter  30 value 17686.183975
## iter  40 value 17583.134425
## iter  50 value 17478.132292
## iter  60 value 17289.869530
## iter  70 value 17282.415486
## iter  80 value 17277.181330
## iter  90 value 17257.235204
## iter 100 value 17225.085646
## final  value 17225.085646 
## stopped after 100 iterations
## # weights:  63
## initial  value 21449.936132 
## iter  10 value 19618.118750
## iter  20 value 19376.853198
## iter  30 value 19281.603134
## iter  40 value 19233.745423
## iter  50 value 19169.413843
## iter  60 value 19112.981165
## iter  70 value 18688.185730
## iter  80 value 17730.605411
## iter  90 value 17505.990262
## iter 100 value 17470.678820
## final  value 17470.678820 
## stopped after 100 iterations
## # weights:  179
## initial  value 22608.272468 
## iter  10 value 18678.474724
## iter  20 value 18561.399248
## iter  30 value 18330.108613
## iter  40 value 18213.228784
## iter  50 value 18150.249865
## iter  60 value 18112.650375
## iter  70 value 18091.692554
## iter  80 value 18058.774332
## iter  90 value 18034.878218
## iter 100 value 18030.313763
## final  value 18030.313763 
## stopped after 100 iterations
## # weights:  295
## initial  value 20520.429631 
## iter  10 value 18151.003471
## iter  20 value 17747.941338
## iter  30 value 17630.755497
## iter  40 value 17396.751655
## iter  50 value 17063.047382
## iter  60 value 16851.984294
## iter  70 value 16373.287656
## iter  80 value 16190.166897
## iter  90 value 16038.486773
## iter 100 value 16000.884914
## final  value 16000.884914 
## stopped after 100 iterations
## # weights:  63
## initial  value 20540.920595 
## iter  10 value 19320.073724
## iter  20 value 18726.628329
## iter  30 value 18469.776584
## iter  40 value 18273.189013
## iter  50 value 18229.444722
## iter  60 value 18169.865549
## iter  70 value 18112.896828
## iter  80 value 18097.031388
## iter  90 value 18011.472703
## iter 100 value 17994.627984
## final  value 17994.627984 
## stopped after 100 iterations
## # weights:  179
## initial  value 21439.357798 
## iter  10 value 19155.804137
## iter  20 value 18447.323566
## iter  30 value 18340.360728
## iter  40 value 17940.468929
## iter  50 value 17713.188720
## iter  60 value 17589.899299
## iter  70 value 17543.399663
## iter  80 value 17523.950361
## iter  90 value 17439.816722
## iter 100 value 17355.961265
## final  value 17355.961265 
## stopped after 100 iterations
## # weights:  295
## initial  value 21848.051783 
## iter  10 value 19184.175959
## iter  20 value 18745.316318
## iter  30 value 18352.551462
## iter  40 value 18127.671427
## iter  50 value 17936.577718
## iter  60 value 17825.540024
## iter  70 value 17665.247442
## iter  80 value 17387.578819
## iter  90 value 17191.661247
## iter 100 value 17140.608920
## final  value 17140.608920 
## stopped after 100 iterations
## # weights:  63
## initial  value 19884.519268 
## iter  10 value 19573.527130
## iter  20 value 19493.164027
## iter  30 value 19340.909507
## iter  40 value 19107.377017
## iter  50 value 18667.280217
## iter  60 value 18296.776350
## iter  70 value 18164.287289
## iter  80 value 18150.481113
## iter  90 value 18102.678846
## iter 100 value 17954.060190
## final  value 17954.060190 
## stopped after 100 iterations
## # weights:  179
## initial  value 21881.451946 
## iter  10 value 19417.691713
## iter  20 value 19256.255679
## iter  30 value 19009.532896
## iter  40 value 18395.291961
## iter  50 value 18269.044727
## iter  60 value 18171.008146
## iter  70 value 17874.814813
## iter  80 value 17789.574723
## iter  90 value 17714.121917
## iter 100 value 17476.499304
## final  value 17476.499304 
## stopped after 100 iterations
## # weights:  295
## initial  value 22153.280937 
## iter  10 value 19213.496218
## iter  20 value 18478.941398
## iter  30 value 18356.522294
## iter  40 value 18293.296360
## iter  50 value 18216.722643
## iter  60 value 17919.165851
## iter  70 value 17493.644519
## iter  80 value 17263.337686
## iter  90 value 17081.654691
## iter 100 value 16975.715506
## final  value 16975.715506 
## stopped after 100 iterations
## # weights:  63
## initial  value 20560.000784 
## iter  10 value 19628.553250
## iter  20 value 19545.649158
## iter  30 value 19353.300341
## iter  40 value 18743.837924
## iter  50 value 18572.745758
## iter  60 value 18435.097169
## iter  70 value 18384.174719
## iter  80 value 18289.180011
## iter  90 value 18218.677540
## iter 100 value 18174.740698
## final  value 18174.740698 
## stopped after 100 iterations
## # weights:  179
## initial  value 21555.064976 
## iter  10 value 18914.123370
## iter  20 value 18839.105906
## iter  30 value 18619.968575
## iter  40 value 18212.886692
## iter  50 value 18102.637173
## iter  60 value 18009.417569
## iter  70 value 17883.287956
## iter  80 value 17809.988316
## iter  90 value 17769.807550
## iter 100 value 17672.014238
## final  value 17672.014238 
## stopped after 100 iterations
## # weights:  295
## initial  value 21902.402344 
## iter  10 value 18741.851293
## iter  20 value 18037.189975
## iter  30 value 17825.880310
## iter  40 value 17667.910338
## iter  50 value 17585.731152
## iter  60 value 17349.917499
## iter  70 value 17219.886103
## iter  80 value 17030.244430
## iter  90 value 16956.190335
## iter 100 value 16919.998371
## final  value 16919.998371 
## stopped after 100 iterations
## # weights:  63
## initial  value 21386.978498 
## iter  10 value 19106.437258
## iter  20 value 18926.403792
## iter  30 value 18543.840210
## iter  40 value 18446.527705
## iter  50 value 18426.799369
## iter  60 value 18418.393994
## iter  70 value 18398.879305
## iter  80 value 18390.515209
## iter  90 value 18297.609524
## iter 100 value 18264.738351
## final  value 18264.738351 
## stopped after 100 iterations
## # weights:  179
## initial  value 21549.585263 
## iter  10 value 18703.410028
## iter  20 value 18518.820040
## iter  30 value 18295.789893
## iter  40 value 18030.359488
## iter  50 value 17960.983429
## iter  60 value 17819.421607
## iter  70 value 17575.685846
## iter  80 value 17290.739772
## iter  90 value 17199.254937
## iter 100 value 16982.081416
## final  value 16982.081416 
## stopped after 100 iterations
## # weights:  295
## initial  value 21200.102126 
## iter  10 value 18428.735281
## iter  20 value 17789.967565
## iter  30 value 17523.114809
## iter  40 value 17343.299893
## iter  50 value 17292.083968
## iter  60 value 17169.971647
## iter  70 value 17114.381131
## iter  80 value 17013.081442
## iter  90 value 16833.656257
## iter 100 value 16769.216585
## final  value 16769.216585 
## stopped after 100 iterations
## # weights:  63
## initial  value 19868.349182 
## iter  10 value 19592.352984
## iter  20 value 19305.451576
## iter  30 value 19215.908166
## iter  40 value 19211.369826
## iter  50 value 19198.853434
## iter  60 value 19196.036209
## iter  70 value 19192.449689
## iter  80 value 19191.481807
## iter  90 value 19188.159324
## iter 100 value 19141.934418
## final  value 19141.934418 
## stopped after 100 iterations
## # weights:  179
## initial  value 26265.420149 
## iter  10 value 19579.297489
## iter  20 value 19472.666197
## iter  30 value 19445.329339
## iter  40 value 19435.987487
## iter  50 value 19429.066272
## iter  60 value 19232.951650
## iter  70 value 19086.426440
## iter  80 value 18946.249042
## iter  90 value 18697.633478
## iter 100 value 18633.526280
## final  value 18633.526280 
## stopped after 100 iterations
## # weights:  295
## initial  value 24672.651553 
## iter  10 value 18968.902565
## iter  20 value 18522.617074
## iter  30 value 18327.922805
## iter  40 value 18057.803552
## iter  50 value 17914.098390
## iter  60 value 17753.627192
## iter  70 value 17673.252634
## iter  80 value 17599.945202
## iter  90 value 17560.758631
## iter 100 value 17495.880725
## final  value 17495.880725 
## stopped after 100 iterations
## # weights:  63
## initial  value 20569.491327 
## iter  10 value 19443.376964
## iter  20 value 19384.084074
## iter  30 value 19309.173112
## iter  40 value 19239.346666
## iter  50 value 19218.038139
## iter  60 value 19061.291479
## iter  70 value 18994.284566
## iter  80 value 18904.424340
## iter  90 value 18885.248990
## iter 100 value 18879.033178
## final  value 18879.033178 
## stopped after 100 iterations
## # weights:  179
## initial  value 23580.794546 
## iter  10 value 18735.538921
## iter  20 value 18623.416102
## iter  30 value 18237.181648
## iter  40 value 18144.532140
## iter  50 value 18127.277391
## iter  60 value 18069.655759
## iter  70 value 17997.359193
## iter  80 value 17838.835138
## iter  90 value 17778.530604
## iter 100 value 17706.486819
## final  value 17706.486819 
## stopped after 100 iterations
## # weights:  295
## initial  value 20941.168971 
## iter  10 value 18407.673490
## iter  20 value 17624.949113
## iter  30 value 17310.577794
## iter  40 value 17078.635064
## iter  50 value 16885.935234
## iter  60 value 16724.611700
## iter  70 value 16515.310657
## iter  80 value 16478.706681
## iter  90 value 16450.171573
## iter 100 value 16372.607109
## final  value 16372.607109 
## stopped after 100 iterations
## # weights:  63
## initial  value 20879.473749 
## iter  10 value 19335.943852
## iter  20 value 19053.131472
## iter  30 value 18935.419401
## iter  40 value 18758.591512
## iter  50 value 18647.736870
## iter  60 value 18556.203216
## iter  70 value 18515.234623
## iter  80 value 18501.266084
## iter  90 value 18457.143668
## iter 100 value 18444.268260
## final  value 18444.268260 
## stopped after 100 iterations
## # weights:  179
## initial  value 24143.612511 
## iter  10 value 19006.437220
## iter  20 value 18746.693194
## iter  30 value 18631.368115
## iter  40 value 18467.328658
## iter  50 value 17899.295826
## iter  60 value 17720.709998
## iter  70 value 17616.193537
## iter  80 value 17506.145318
## iter  90 value 17447.343688
## iter 100 value 17349.727937
## final  value 17349.727937 
## stopped after 100 iterations
## # weights:  295
## initial  value 21812.015666 
## iter  10 value 18925.684969
## iter  20 value 18704.922103
## iter  30 value 18323.337346
## iter  40 value 18074.448291
## iter  50 value 18040.380379
## iter  60 value 18022.942940
## iter  70 value 18007.830387
## iter  80 value 17992.462031
## iter  90 value 17954.493445
## iter 100 value 17801.825805
## final  value 17801.825805 
## stopped after 100 iterations
## # weights:  63
## initial  value 19867.548087 
## iter  10 value 19228.910631
## iter  20 value 18549.175220
## iter  30 value 18361.985601
## iter  40 value 18305.581709
## iter  50 value 18145.446854
## iter  60 value 18091.161167
## iter  70 value 18028.448257
## iter  80 value 17975.071353
## iter  90 value 17943.352729
## iter 100 value 17929.618788
## final  value 17929.618788 
## stopped after 100 iterations
## # weights:  179
## initial  value 21588.664674 
## iter  10 value 18807.907862
## iter  20 value 18501.993976
## iter  30 value 18166.002611
## iter  40 value 18071.573774
## iter  50 value 17903.836544
## iter  60 value 17763.955271
## iter  70 value 17618.871814
## iter  80 value 17380.955980
## iter  90 value 17306.603322
## iter 100 value 17273.039141
## final  value 17273.039141 
## stopped after 100 iterations
## # weights:  295
## initial  value 23218.683298 
## iter  10 value 18721.465334
## iter  20 value 18455.012774
## iter  30 value 17732.185509
## iter  40 value 17575.728287
## iter  50 value 17458.648474
## iter  60 value 17183.352798
## iter  70 value 17064.051834
## iter  80 value 16792.403792
## iter  90 value 16443.905887
## iter 100 value 16344.710632
## final  value 16344.710632 
## stopped after 100 iterations
## # weights:  63
## initial  value 21146.175683 
## iter  10 value 18959.456435
## iter  20 value 18898.345155
## iter  30 value 18691.773148
## iter  40 value 18512.538990
## iter  50 value 18476.248082
## iter  60 value 18440.819416
## iter  70 value 18439.698350
## iter  80 value 18430.897536
## iter  90 value 18423.091369
## iter 100 value 18419.215566
## final  value 18419.215566 
## stopped after 100 iterations
## # weights:  179
## initial  value 21091.797425 
## iter  10 value 19275.703484
## iter  20 value 19151.344992
## iter  30 value 19081.282532
## iter  40 value 18895.540487
## iter  50 value 18798.508027
## iter  60 value 18512.860323
## iter  70 value 18119.777884
## iter  80 value 18036.184526
## iter  90 value 17900.751273
## iter 100 value 17688.466185
## final  value 17688.466185 
## stopped after 100 iterations
## # weights:  295
## initial  value 22793.050377 
## iter  10 value 18854.357016
## iter  20 value 18444.806128
## iter  30 value 18189.697085
## iter  40 value 18027.227648
## iter  50 value 17702.060853
## iter  60 value 17558.480966
## iter  70 value 17319.424958
## iter  80 value 17149.375810
## iter  90 value 17117.067128
## iter 100 value 17106.462870
## final  value 17106.462870 
## stopped after 100 iterations
## # weights:  63
## initial  value 20765.369870 
## iter  10 value 19305.523886
## iter  20 value 19196.493666
## iter  30 value 18889.440482
## iter  40 value 18815.302970
## iter  50 value 18779.582545
## iter  60 value 18752.603881
## iter  70 value 18568.929582
## iter  80 value 18135.346549
## iter  90 value 18122.754648
## iter 100 value 18111.922585
## final  value 18111.922585 
## stopped after 100 iterations
## # weights:  179
## initial  value 21360.453415 
## iter  10 value 18649.841653
## iter  20 value 18294.164939
## iter  30 value 17873.076373
## iter  40 value 17725.681109
## iter  50 value 17601.520738
## iter  60 value 17362.189792
## iter  70 value 16994.139618
## iter  80 value 16885.486402
## iter  90 value 16785.115117
## iter 100 value 16615.678236
## final  value 16615.678236 
## stopped after 100 iterations
## # weights:  295
## initial  value 21375.839918 
## iter  10 value 19023.775886
## iter  20 value 18498.269062
## iter  30 value 18362.835440
## iter  40 value 18186.704594
## iter  50 value 18072.250764
## iter  60 value 17927.546475
## iter  70 value 17876.576181
## iter  80 value 17840.957247
## iter  90 value 17759.670700
## iter 100 value 17707.146672
## final  value 17707.146672 
## stopped after 100 iterations
## # weights:  63
## initial  value 20726.237805 
## iter  10 value 19079.482922
## iter  20 value 18711.096437
## iter  30 value 18655.224054
## iter  40 value 18629.979298
## iter  50 value 18519.591769
## iter  60 value 18315.212114
## iter  70 value 18290.917976
## iter  80 value 18251.220015
## iter  90 value 18241.883967
## iter 100 value 18237.949457
## final  value 18237.949457 
## stopped after 100 iterations
## # weights:  179
## initial  value 21449.383653 
## iter  10 value 19429.950862
## iter  20 value 18744.611586
## iter  30 value 18528.136534
## iter  40 value 18402.424498
## iter  50 value 18148.110591
## iter  60 value 18078.770140
## iter  70 value 17995.433346
## iter  80 value 17822.419108
## iter  90 value 17664.802874
## iter 100 value 17454.072088
## final  value 17454.072088 
## stopped after 100 iterations
## # weights:  295
## initial  value 23055.410266 
## iter  10 value 19380.906405
## iter  20 value 19262.076103
## iter  30 value 19066.905519
## iter  40 value 18409.243326
## iter  50 value 18246.882171
## iter  60 value 18139.356337
## iter  70 value 18105.251912
## iter  80 value 17834.255678
## iter  90 value 17773.347815
## iter 100 value 17712.335105
## final  value 17712.335105 
## stopped after 100 iterations
## # weights:  63
## initial  value 20570.275907 
## iter  10 value 19062.881465
## iter  20 value 18761.351287
## iter  30 value 18629.039558
## iter  40 value 18611.146054
## iter  50 value 18487.715607
## iter  60 value 18207.023015
## iter  70 value 17791.135440
## iter  80 value 17482.973513
## iter  90 value 17273.179958
## iter 100 value 16984.595181
## final  value 16984.595181 
## stopped after 100 iterations
## # weights:  179
## initial  value 20123.825532 
## iter  10 value 18762.714597
## iter  20 value 18385.173682
## iter  30 value 18093.254242
## iter  40 value 17985.751857
## iter  50 value 17942.552055
## iter  60 value 17884.353966
## iter  70 value 17750.267561
## iter  80 value 17598.651123
## iter  90 value 17298.543422
## iter 100 value 17011.447561
## final  value 17011.447561 
## stopped after 100 iterations
## # weights:  295
## initial  value 20630.642490 
## iter  10 value 18835.613929
## iter  20 value 18195.180663
## iter  30 value 18141.773752
## iter  40 value 18113.614800
## iter  50 value 17952.802322
## iter  60 value 17744.925001
## iter  70 value 17686.835862
## iter  80 value 17564.834608
## iter  90 value 17448.155313
## iter 100 value 17359.462419
## final  value 17359.462419 
## stopped after 100 iterations
## # weights:  63
## initial  value 20492.714374 
## iter  10 value 19572.627733
## iter  20 value 19552.200341
## iter  30 value 19546.482710
## iter  40 value 19472.938662
## iter  50 value 19172.458673
## iter  60 value 19021.756738
## iter  70 value 19015.995000
## iter  80 value 19012.479517
## iter  90 value 18889.314439
## iter 100 value 18487.332035
## final  value 18487.332035 
## stopped after 100 iterations
## # weights:  179
## initial  value 22312.891881 
## iter  10 value 19422.786417
## iter  20 value 19017.272340
## iter  30 value 18522.471642
## iter  40 value 18232.583231
## iter  50 value 18134.455619
## iter  60 value 18044.274074
## iter  70 value 17934.975700
## iter  80 value 17649.232367
## iter  90 value 17618.010240
## iter 100 value 17597.874917
## final  value 17597.874917 
## stopped after 100 iterations
## # weights:  295
## initial  value 22042.042225 
## iter  10 value 18512.947870
## iter  20 value 18104.461427
## iter  30 value 17910.382368
## iter  40 value 17747.405761
## iter  50 value 17479.373163
## iter  60 value 17356.401668
## iter  70 value 17182.710866
## iter  80 value 17018.852072
## iter  90 value 16873.144756
## iter 100 value 16805.288633
## final  value 16805.288633 
## stopped after 100 iterations
## # weights:  63
## initial  value 20002.063017 
## iter  10 value 18921.282164
## iter  20 value 18501.329663
## iter  30 value 18411.402091
## iter  40 value 18321.916733
## iter  50 value 18240.756694
## iter  60 value 18218.995323
## iter  70 value 18199.863901
## iter  80 value 18190.534823
## iter  90 value 18188.008448
## iter 100 value 18165.467044
## final  value 18165.467044 
## stopped after 100 iterations
## # weights:  179
## initial  value 19825.826295 
## iter  10 value 19195.024702
## iter  20 value 18864.191990
## iter  30 value 18655.710377
## iter  40 value 18435.217510
## iter  50 value 18267.087656
## iter  60 value 18171.744427
## iter  70 value 18108.206753
## iter  80 value 17916.193705
## iter  90 value 17778.806168
## iter 100 value 17699.512944
## final  value 17699.512944 
## stopped after 100 iterations
## # weights:  295
## initial  value 20668.601562 
## iter  10 value 18402.513536
## iter  20 value 17951.578116
## iter  30 value 17589.180424
## iter  40 value 17301.228401
## iter  50 value 17084.379450
## iter  60 value 16799.951561
## iter  70 value 16678.341582
## iter  80 value 16556.169439
## iter  90 value 16496.761716
## iter 100 value 16470.961753
## final  value 16470.961753 
## stopped after 100 iterations
## # weights:  63
## initial  value 21143.000105 
## iter  10 value 19426.566745
## iter  20 value 19421.547892
## iter  30 value 19390.836736
## iter  40 value 19374.002067
## iter  50 value 19353.476272
## iter  60 value 19346.821805
## iter  70 value 19316.477061
## iter  80 value 19243.973777
## iter  90 value 19199.898002
## iter 100 value 19084.991588
## final  value 19084.991588 
## stopped after 100 iterations
## # weights:  179
## initial  value 21005.247642 
## iter  10 value 18282.385612
## iter  20 value 18177.654326
## iter  30 value 17976.807935
## iter  40 value 17822.106112
## iter  50 value 17300.747078
## iter  60 value 17263.279256
## iter  70 value 17185.528319
## iter  80 value 17165.634954
## iter  90 value 17033.925192
## iter 100 value 16939.144580
## final  value 16939.144580 
## stopped after 100 iterations
## # weights:  295
## initial  value 31311.918362 
## iter  10 value 18654.563828
## iter  20 value 18207.409041
## iter  30 value 17834.796418
## iter  40 value 17566.537291
## iter  50 value 17490.827176
## iter  60 value 17296.643269
## iter  70 value 17223.202290
## iter  80 value 17169.836209
## iter  90 value 17139.966242
## iter 100 value 17115.015152
## final  value 17115.015152 
## stopped after 100 iterations
## # weights:  63
## initial  value 20332.165640 
## iter  10 value 19216.934854
## iter  20 value 18890.314595
## iter  30 value 18605.165489
## iter  40 value 18464.614544
## iter  50 value 18447.226179
## iter  60 value 18396.276106
## iter  70 value 18385.203481
## iter  80 value 18373.923942
## iter  90 value 18372.778029
## iter 100 value 18372.685849
## final  value 18372.685849 
## stopped after 100 iterations
## # weights:  179
## initial  value 22020.534822 
## iter  10 value 18574.015705
## iter  20 value 18142.227610
## iter  30 value 17584.769274
## iter  40 value 17108.368092
## iter  50 value 16649.138345
## iter  60 value 16415.854798
## iter  70 value 16268.818798
## iter  80 value 16190.682227
## iter  90 value 16037.976866
## iter 100 value 15940.168590
## final  value 15940.168590 
## stopped after 100 iterations
## # weights:  295
## initial  value 29853.197214 
## iter  10 value 19638.539676
## iter  20 value 19198.405772
## iter  30 value 19113.832210
## iter  40 value 19072.761038
## iter  50 value 19030.261950
## iter  60 value 18984.163209
## iter  70 value 18959.091649
## iter  80 value 18930.344996
## iter  90 value 18905.879242
## iter 100 value 18889.882296
## final  value 18889.882296 
## stopped after 100 iterations
## # weights:  63
## initial  value 20429.465349 
## iter  10 value 19804.103447
## iter  20 value 19452.182377
## iter  30 value 19224.775336
## iter  40 value 19217.691366
## iter  50 value 19182.185494
## iter  60 value 19005.195729
## iter  70 value 18957.502294
## iter  80 value 18952.005117
## iter  90 value 18947.813844
## iter 100 value 18946.084926
## final  value 18946.084926 
## stopped after 100 iterations
## # weights:  179
## initial  value 20929.788423 
## iter  10 value 19072.910208
## iter  20 value 18904.971507
## iter  30 value 18148.294390
## iter  40 value 17864.712826
## iter  50 value 17610.354590
## iter  60 value 17479.544127
## iter  70 value 17441.527370
## iter  80 value 17385.425378
## iter  90 value 17309.070725
## iter 100 value 17281.628004
## final  value 17281.628004 
## stopped after 100 iterations
## # weights:  295
## initial  value 21155.176135 
## iter  10 value 18881.573943
## iter  20 value 18089.512029
## iter  30 value 17593.421256
## iter  40 value 17157.772499
## iter  50 value 16848.937682
## iter  60 value 16688.698818
## iter  70 value 16586.899007
## iter  80 value 16365.530343
## iter  90 value 16313.024001
## iter 100 value 16281.806317
## final  value 16281.806317 
## stopped after 100 iterations
## # weights:  63
## initial  value 20472.276343 
## iter  10 value 19711.658068
## iter  20 value 19664.349953
## iter  30 value 19596.493011
## iter  40 value 19314.617843
## iter  50 value 18887.107132
## iter  60 value 18713.599950
## iter  70 value 18693.902673
## iter  80 value 18623.233111
## iter  90 value 18593.482903
## iter 100 value 18592.039089
## final  value 18592.039089 
## stopped after 100 iterations
## # weights:  179
## initial  value 22310.972094 
## iter  10 value 18423.555060
## iter  20 value 18129.527012
## iter  30 value 18036.584808
## iter  40 value 17990.520165
## iter  50 value 17934.500888
## iter  60 value 17832.182178
## iter  70 value 17728.532826
## iter  80 value 17615.037901
## iter  90 value 17321.567551
## iter 100 value 17196.495152
## final  value 17196.495152 
## stopped after 100 iterations
## # weights:  295
## initial  value 23067.755544 
## iter  10 value 18910.832493
## iter  20 value 18692.461326
## iter  30 value 18626.476251
## iter  40 value 18466.001990
## iter  50 value 18157.650780
## iter  60 value 17997.277568
## iter  70 value 17920.672962
## iter  80 value 17889.243746
## iter  90 value 17873.331859
## iter 100 value 17801.525996
## final  value 17801.525996 
## stopped after 100 iterations
## # weights:  63
## initial  value 21884.743348 
## iter  10 value 19489.994312
## iter  20 value 19400.353849
## iter  30 value 19359.421243
## iter  40 value 19354.036364
## iter  50 value 19344.435995
## iter  60 value 19335.170146
## iter  70 value 19333.899113
## iter  80 value 19333.250794
## iter  90 value 19333.109033
## iter 100 value 19333.049490
## final  value 19333.049490 
## stopped after 100 iterations
## # weights:  179
## initial  value 21781.540305 
## iter  10 value 18799.730645
## iter  20 value 18168.516986
## iter  30 value 18012.126466
## iter  40 value 17876.108157
## iter  50 value 17684.808839
## iter  60 value 17657.391610
## iter  70 value 17424.647587
## iter  80 value 17382.038985
## iter  90 value 17362.099620
## iter 100 value 17302.710475
## final  value 17302.710475 
## stopped after 100 iterations
## # weights:  295
## initial  value 21164.497410 
## iter  10 value 18482.657434
## iter  20 value 18189.080670
## iter  30 value 17471.744795
## iter  40 value 17085.055579
## iter  50 value 16899.611808
## iter  60 value 16748.350125
## iter  70 value 16595.120776
## iter  80 value 16454.246689
## iter  90 value 16356.677656
## iter 100 value 16265.514018
## final  value 16265.514018 
## stopped after 100 iterations
## # weights:  63
## initial  value 20948.177347 
## iter  10 value 18808.441963
## iter  20 value 18483.134549
## iter  30 value 18165.132518
## iter  40 value 18044.739537
## iter  50 value 17901.326888
## iter  60 value 17707.807460
## iter  70 value 17591.988360
## iter  80 value 17469.330138
## iter  90 value 17451.207580
## iter 100 value 17437.545611
## final  value 17437.545611 
## stopped after 100 iterations
## # weights:  179
## initial  value 22249.947111 
## iter  10 value 19437.224112
## iter  20 value 18815.785418
## iter  30 value 18109.737845
## iter  40 value 17859.775912
## iter  50 value 17668.938061
## iter  60 value 17209.516023
## iter  70 value 16945.939997
## iter  80 value 16836.742802
## iter  90 value 16801.743336
## iter 100 value 16620.511055
## final  value 16620.511055 
## stopped after 100 iterations
## # weights:  295
## initial  value 24904.876961 
## iter  10 value 19179.165562
## iter  20 value 18354.337644
## iter  30 value 17974.382163
## iter  40 value 17430.880458
## iter  50 value 17155.301537
## iter  60 value 16987.651636
## iter  70 value 16771.087886
## iter  80 value 16494.497496
## iter  90 value 16361.665688
## iter 100 value 16259.619223
## final  value 16259.619223 
## stopped after 100 iterations
## # weights:  63
## initial  value 20498.774986 
## iter  10 value 19555.921247
## iter  20 value 19499.209622
## iter  30 value 19294.228985
## iter  40 value 19064.235158
## iter  50 value 18805.218863
## iter  60 value 18191.271622
## iter  70 value 18153.703315
## iter  80 value 18131.805673
## iter  90 value 18129.353925
## iter 100 value 18009.675178
## final  value 18009.675178 
## stopped after 100 iterations
## # weights:  179
## initial  value 24387.772530 
## iter  10 value 19505.265476
## iter  20 value 18730.175292
## iter  30 value 18277.863971
## iter  40 value 18230.530417
## iter  50 value 18207.582257
## iter  60 value 18173.974660
## iter  70 value 18134.252751
## iter  80 value 18121.869397
## iter  90 value 18090.176830
## iter 100 value 17986.616284
## final  value 17986.616284 
## stopped after 100 iterations
## # weights:  295
## initial  value 22243.829166 
## iter  10 value 18052.295038
## iter  20 value 17527.987606
## iter  30 value 17093.150218
## iter  40 value 16968.540074
## iter  50 value 16887.586721
## iter  60 value 16792.042457
## iter  70 value 16735.563405
## iter  80 value 16639.735395
## iter  90 value 16527.206555
## iter 100 value 16483.559244
## final  value 16483.559244 
## stopped after 100 iterations
## # weights:  63
## initial  value 20127.239781 
## iter  10 value 19595.657503
## iter  20 value 19587.792514
## iter  30 value 19570.654329
## iter  40 value 19568.514245
## iter  50 value 19566.557668
## iter  60 value 19565.769806
## iter  70 value 19558.848620
## iter  80 value 19558.370194
## iter  90 value 19558.222741
## iter 100 value 19558.193787
## final  value 19558.193787 
## stopped after 100 iterations
## # weights:  179
## initial  value 22833.465657 
## iter  10 value 19332.297515
## iter  20 value 19307.772463
## iter  30 value 19216.284370
## iter  40 value 19024.820861
## iter  50 value 18999.323307
## iter  60 value 18977.056859
## iter  70 value 18919.691068
## iter  80 value 18780.862514
## iter  90 value 18654.787421
## iter 100 value 18643.752245
## final  value 18643.752245 
## stopped after 100 iterations
## # weights:  295
## initial  value 21526.354484 
## iter  10 value 18626.946566
## iter  20 value 18185.372008
## iter  30 value 18002.149656
## iter  40 value 17761.549305
## iter  50 value 17611.387401
## iter  60 value 17394.156004
## iter  70 value 17195.321737
## iter  80 value 17031.411985
## iter  90 value 16963.528431
## iter 100 value 16927.279894
## final  value 16927.279894 
## stopped after 100 iterations
## # weights:  295
## initial  value 28609.172379 
## iter  10 value 21272.535417
## iter  20 value 20766.646089
## iter  30 value 20306.386959
## iter  40 value 19867.750079
## iter  50 value 19615.258124
## iter  60 value 19149.815433
## iter  70 value 19003.646430
## iter  80 value 18882.208232
## iter  90 value 18683.633487
## iter 100 value 18591.696316
## final  value 18591.696316 
## stopped after 100 iterations
```

```r
print(fit_nn, digits = 4)
```

```
## Neural Network 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12363, 12364, 12363, 12363, 12363, ... 
## Resampling results across tuning parameters:
## 
##   size  decay  Accuracy  Kappa 
##   1     0e+00  0.3295    0.1286
##   1     1e-04  0.3331    0.1227
##   1     1e-01  0.3340    0.1316
##   3     0e+00  0.3834    0.2086
##   3     1e-04  0.3776    0.2082
##   3     1e-01  0.3785    0.2052
##   5     0e+00  0.4184    0.2600
##   5     1e-04  0.3872    0.2145
##   5     1e-01  0.4047    0.2503
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were size = 5 and decay = 0.
```

Now let's look at the prediction on the validation set..


```r
predict_nn_validating <- predict(fit_nn, myvalidating)
conf_nn_validating <- confusionMatrix(myvalidating$classe, predict_nn_validating)
conf_nn_validating
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1143   27  229  272    3
##          B  223  253  264  394    5
##          C  408   39  479   91    9
##          D  142   18  191  605    8
##          E  162   79  300  496   45
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4291          
##                  95% CI : (0.4164, 0.4418)
##     No Information Rate : 0.3531          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.276           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5500  0.60817  0.32741   0.3256 0.642857
## Specificity            0.8605  0.83800  0.87630   0.9109 0.821668
## Pos Pred Value         0.6828  0.22212  0.46686   0.6276 0.041590
## Neg Pred Value         0.7780  0.96566  0.79749   0.7454 0.994795
## Prevalence             0.3531  0.07069  0.24860   0.3157 0.011895
## Detection Rate         0.1942  0.04299  0.08139   0.1028 0.007647
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638 0.183857
## Balanced Accuracy      0.7053  0.72308  0.60185   0.6182 0.732263
```
### Neural Networks gives an accuracy of .4116 and looking at the balanced accuracy for the various classes it ranges from .40 - .69


Now lets see what Random forest can do.

```r
fit_rf <- train(classe ~ ., data = myTraining, method = "rf", 
                   trControl = foldcontrol)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.3.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
print(fit_rf, digits = 4)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12363, 12364, 12365, 12362, 12363, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##    2    0.9913    0.9889
##   27    0.9915    0.9892
##   52    0.9858    0.9820
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

Now let's look at the prediction on the validation set..


```r
predict_rf_validating <- predict(fit_rf, myvalidating)
conf_rf_validating <- confusionMatrix(myvalidating$classe, predict_rf_validating)
conf_rf_validating
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B   10 1128    1    0    0
##          C    0    8 1013    5    0
##          D    0    1    9  954    0
##          E    0    0    2    3 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.286           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9941   0.9912   0.9883   0.9917   1.0000
## Specificity            0.9998   0.9977   0.9973   0.9980   0.9990
## Pos Pred Value         0.9994   0.9903   0.9873   0.9896   0.9954
## Neg Pred Value         0.9976   0.9979   0.9975   0.9984   1.0000
## Prevalence             0.2860   0.1934   0.1742   0.1635   0.1830
## Detection Rate         0.2843   0.1917   0.1721   0.1621   0.1830
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9969   0.9944   0.9928   0.9948   0.9995
```
For this dataset, random forest method is way better than classification tree method. The accuracy rate is 0.9927. 

##Observation:
The Random forest computationally was very expensive and was not efficient.


## Prediction on the Testing Data:

Having identified Random forest is better of the two - we can now run it on the test data. 

Let now predict the outcome on the testing set..


```r
predict_rf_testing <-predict(fit_rf, testing)
predict_rf_testing
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


