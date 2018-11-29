### Name :  Soham Bhalerao
### Data_Set : Titanic Survival
### Date : 7/2/2018

rm(list = ls())
library(data.table)
## Decision trees and DT graohics
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(maptree)
library(dplyr)
  

## Importing Training and Test Data ###
titanic_train <- fread("train.csv")
titanic_test <- fread("test.csv")

View(titanic_train)
View(titanic_test)

# Structure of data
str(titanic_train) 
str(titanic_test) 


# Look at he number of People
table(titanic_train$Survived)
prop.table(table(titanic_train$Survived))

# Predicting on test data that all passengers died, created a new column Survived.
titanic_test$Survived <- rep(0,418) 

# First Submission, creating a csv file with two columns
submit <- data.frame(PassengerId = titanic_test$PassengerId,Survived = titanic_test$Survived)
write.csv(submit, file = "AllDied", row.names = FALSE)

# Checking the differnece in survival rate of men and women.
table(titanic_train$Sex)
summary(titanic_train$Sex)

# Converting sex to a factor, female is TRUE
titanic_train$Sex <- as.factor(titanic_train$Sex == "female")
titanic_test$Sex <- as.factor(titanic_test$Sex == "female")

prop.table(table(titanic_train$Sex))
prop.table(table(titanic_train$Survived))

# Proportion of males and females who survived.
prop.table(table(titanic_train$Sex, titanic_train$Survived),1)

# Using this prediction and updating in test dataset.
titanic_test$Survived <- 0
titanic_test$Survived[titanic_test$Sex == "female"] <- 1
prop.table(table(titanic_test$Survived))

# Writing 2nd prediction to the csv Submission2.
submit <- data.frame(PassengerId = titanic_test$PassengerId,Survived = titanic_test$Survived)
write.csv(submit, file = "Submission2", row.names = FALSE)


## Create a variable child in training data set, where age less 18.
summary(titanic_train$Age)
titanic_train$Child <- 0  
titanic_train$Child[titanic_train$Age < 18] <- 1

##  Using aggregate function to know survival proportions depending on children, TRUE is Sex is female

aggregate(Survived ~ Child + Sex , data = titanic_train, FUN = function(x) {sum(x)/length(x)})

## Create multiple groups depending on Fare.
titanic_train$Fare2 <- "30+"
titanic_train$Fare2[titanic_train$Fare < 30 & titanic_train$Fare >= 20] <- "20-30"

titanic_train$Fare2[titanic_train$Fare < 20 & titanic_train$Fare >= 10] <- "10-20"
titanic_train$Fare2[titanic_train$Fare < 10] <- "<10"

aggregate(Survived ~ Fare2 + Pclass + Sex, data = titanic_train, FUN = function(x) {sum(x)/length(x)})

# Using this prediction and updating in test dataset.
titanic_test$Survived <- 0
titanic_test$Survived[titanic_test$Sex == "female"] <- 1
titanic_test$Survived[titanic_test$Sex == "female" & titanic_test$Pclass == 3 & titanic_test$Fare >= 20] <- 0

# Writing 3nd prediction to the csv Submission3.
submit <- data.frame(PassengerId = titanic_test$PassengerId,Survived = titanic_test$Survived)
write.csv(submit, file = "Submission3", row.names = FALSE)


##  Using decision trees for prediction using rpart , method class will be used as CART and for continuous anova)

model1 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = titanic_train, method = "class")
#summary(model1)

## Plotting to see how decision tree looks.
plot(model1)
text(model1)

## Fancy Graphics for DT's

fancyRpartPlot(model1, main = "Decision Tress where Sex is False if male", cex = .6)

##### Predicting based on DT and submitting Prediction 4 ######

Predictions <- predict(model1, titanic_test , type = "class")
submit <- data.frame(PassengerId = titanic_test$PassengerId,Survived = Predictions)
write.csv(submit, file = "Submission4", row.names = FALSE)

### Feature engineering ###
titanic_train$Name[1]

### Comnining test and training data #####

titanic_test$Survived <- NA ## because the data does not have same number of data points.
titanic_test$Child <- NA 
titanic_test$Fare2 <- NA
combination <- rbind(titanic_train,titanic_test) ## Should have same number of columns so we created CHild and Fare2 in Testing dataset.

#### String split using the name for each row using sapply, normal apply will not be used.

combination$Title <- sapply(combination$Name, FUN=function(x) {strsplit(x,split = "[,.]")[[1]][[2]]}) 

head(combination$Title)

### Remove spaces from the Title.
combination$Title <- sub(' ','',combination$Title)
head(combination$Title)
table(combination$Title)

#### Combining Titles using %in% which checks if we have a value in the vector
combination$Title[combination$Title %in% c("Mlle","Mme")] <- "Mme"
combination$Title[combination$Title %in% c("Capt","Don","Major","Sir")] <- "Sir"
combination$Title[combination$Title %in% c("Dona","Lady","the Countess","Jonkheer")] <- "Lady"
table(combination$Title)

## Converting to a Title to factor

combination$Title <- factor(combination$Title)

## Combining All the family members

combination$FamilySize <- combination$SibSp + combination$Parch + 1

## Getting the family Surnames from the name
combination$SurnameName <- sapply(combination$Name , FUN = function(x) {strsplit(x,split = "[,.]")[[1]][[1]]})

## Combining Family Size and Surname

combination$FamilyID <- paste(as.character(combination$FamilySize),combination$SurnameName, sep = '')

### KNocking out family size witn less 2

combination$FamilyID[combination$FamilySize <= 2] <- "Small"

### Creating a data frame of FamilyID table.
famID <- data.frame(table(combination$FamilyID))
famID <- famID[famID$Freq <= 2,]

### Finally knocking out families with size less tha 2

combination$FamilyID[combination$FamilyID %in% famID$Var1] <- "Small"
combination$FamilyID <- factor(combination$FamilyID)


## Creating test and train variables depending on sizes of the two datasets

train <- combination[1:891,]
test <- combination[892:1309,]


## Predications
model2 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,data=train,method="class")
fancyRpartPlot(model2)

### Predications

Predictions <- predict(model2, test , type = "class")
submit <- data.frame(PassengerId = titanic_test$PassengerId,Survived = Predictions)
write.csv(submit, file = "Submission5", row.names = FALSE)


### Random Forest
summary(combination$Age)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                  data=combination[!is.na(combination$Age),], 
                  method="anova")
combination$Age[is.na(combination$Age)] <- predict(Agefit, combination[is.na(combination$Age),])

which(combination$Embarked == '')
combination$Embarked[c(62,830)] = "S"
combination$Embarked <- factor(combination$Embarked)

summary(combination$Fare)
which(is.na(combination$Fare))
combination$Fare[1044] <- median(combination$Fare, na.rm=TRUE)
combination$FamilyID2 <- combination$FamilyID
combination$FamilyID2 <- as.character(combination$FamilyID2)
combination$FamilyID2[combination$FamilySize <= 3] <- 'Small'
combination$FamilyID2 <- as.factor(combination$FamilyID2)
install.packages("randomForest")
train <- combination[1:891,]
test <- combination[892:1309,]
library(randomForest)

library(party)
set.seed(415)
train$Survived <- as.factor(train$Survived)

train$Embarked <- as.factor(train$Embarked)
levels(test$Survived) <-  levels(train$Survived)
levels(test$Embarked) <- levels(train$Embarked)


fit <- cforest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare2 + Embarked + Title + FamilySize + FamilyID,data = train, controls=cforest_unbiased(ntree=2000, mtry=3))



Prediction <- predict(fit,test, OOB = TRUE, type = "response")

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "Submission7", row.names = FALSE)


