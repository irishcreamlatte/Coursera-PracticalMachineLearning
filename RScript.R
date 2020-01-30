###############
## PACKAGES ###
###############

library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

###############
## DOWNLOAD ###
###############

training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                     na.strings = c("NA", "#DIV/0!", ""))
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                    na.strings = c("NA", "#DIV/0!", ""))

###############
##### EDA #####
###############

dim(training)
names(training)
summary(training)
glimpse(training) 
head(training)

table(training$classe)

training %>% group_by(classe) %>% 
    summarize(n = n()) %>% 
    ggplot(aes(classe, n, fill = classe)) + 
    geom_bar(stat = "identity") +
    labs(title = "Total by Classe", x = "Classe", y = "Number")

## Remove NAs ## 

training <- training[,colSums(is.na(training)) == 0]
testing <- testing[,colSums(is.na(testing)) == 0]

## Retain only measurement variables ## 

training <- training %>% select(roll_belt:classe)
testing <- testing %>% select(roll_belt:magnet_forearm_z)

## Define train and test sets for cross validation # 

set.seed(123, sample.kind = "Rounding")
train_index <- createDataPartition(training$classe, p = 0.6, list = FALSE)
cv_train <- training %>% slice(train_index)
cv_test <- training %>% slice(-train_index)

###############
#### TREE #####
###############

set.seed(1234, sample.kind = "Rounding")
rpart_fit <- rpart(classe ~ ., data = cv_train, method = "class")
rpart_pred <- predict(rpart_fit, cv_test, type = "class")
confusionMatrix(rpart_pred, cv_test$classe)$overall[1]

###############
##### KNN #####
###############

set.seed(1234, sample.kind = "Rounding")
knn_fit <- knn3(classe ~ ., data = cv_train, type = "class")

pred_knn <- predict(knn_fit, cv_test, type = "class")

confusionMatrix(pred_knn, cv_test$classe)$overall[1]

###############
#RANDOM FOREST#
###############

set.seed(1234, sample.kind = "Rounding")

rf_fit <- randomForest(classe ~ ., data = cv_train)
pred_rf <- predict(rf_fit, cv_test)
confusionMatrix(pred_rf, cv_test$classe)$overall[1]

###############
##### PCA #####
###############

set.seed(1234, sample.kind = "Rounding")
x <- cv_train[,1:52] %>% as.matrix()
pca <- prcomp(x)
summary(pca)

data.frame(pca$x[,1:2], Classe = cv_train$classe) %>% 
    ggplot(aes(PC1, PC2, fill = Classe)) + 
    geom_point(cex = 3, pch = 21) + 
    coord_fixed(ratio = 1)

data.frame(pca$x[,8:9], Classe = cv_train$classe) %>% 
    ggplot(aes(PC8, PC9, fill = Classe)) + 
    geom_point(cex = 3, pch = 21) + 
    coord_fixed(ratio = 1)


x_train <- pca$x[,1:9]
y <- factor(cv_train$classe)
fit_pca <- knn3(x_train, y)


col_means <- colMeans(cv_test[,1:52])
x_test <- sweep(as.matrix(cv_test[,1:52]), 2, col_means) %*%
    pca$rotation
x_test <- x_test[,1:9]

pred_pca_knn <- predict(fit_pca, x_test, type = "class")
confusionMatrix(pred_pca_knn, cv_test$classe)$overall[1]

fit_pca_rf <- randomForest(x_train, y)
pred_pca_rf <- predict(fit_pca_rf, x_test, type = "class")
confusionMatrix(pred_pca_rf, cv_test$classe)$overall[1]

###############
##FINAL MODEL##
###############

prediction <- predict(rf_fit, testing)
prediction
