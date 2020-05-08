#The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
#For more details, consult the reference [Cortez et al., 2009]. Due to privacy and logistic issues,
#only physicochemical (inputs) and sensory (the output) variables are available 
#(e.g. there is no data about grape types, wine brand, wine selling price, etc.).

#These datasets can be viewed as classification or regression tasks. 
#The classes are ordered and not balanced (e.g. there are much more normal wines than excellent 
#or poor ones).

#This dataset is also available from the UCI machine learning repository,
#https://archive.ics.uci.edu/ml/datasets/wine+quality 

#For more information, read [Cortez et al., 2009].

#Input variables (based on physicochemical tests):
  
#1 - fixed acidity 2 - volatile acidity 3 - citric acid 4 - residual sugar 5 - chlorides 6 - free sulfur dioxide

#7 - total sulfur dioxide 8 - density 9 - pH  10 - sulphates  11 - alcohol

#Output variable (based on sensory data):  12 - quality (score between 0 and 10)

library(ggplot2)
library(dummies)
library(rpart)
library(rpart.plot)
library(caTools)
library(e1071)
library(class)

df <- read.csv("C:/Users/JOGESH MISHRA/Downloads/ML DATA/winequality-red.csv", header = TRUE)
View(df)

summary(df)

pairs(~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide
      +total.sulfur.dioxide+density+pH+sulphates+alcohol,data=df,pch =21
      ,bg=c("magenta","yellow","red","green","pink","blue","orange")[unclass(df$quality)])

# Individual Scattter Plots of one variable with other can be plotted using 
# ggplot(df(x=,y=))+geom.jitter(aes(color=quality))
# where x and y are variables from data df

barplot(table(df$quality))

barplot(table((df$density)))
barplot(table(df$pH))

boxplot(df[,1:12])


#OUTLIER TREATMENT 

hv<- quantile(df$fixed.acidity,0.96)
df$fixed.acidity[df$fixed.acidity >= hv] <- hv
boxplot(df$fixed.acidity)

hv<- quantile(df$volatile.acidity,0.98)
df$volatile.acidity[df$volatile.acidity > hv] <- hv
boxplot(df$volatile.acidity)

hv<- quantile(df$citric.acid,0.99)
df$citric.acid[df$citric.acid > hv] <- hv
boxplot(df$citric.acid)

hv<- quantile(df$residual.sugar,0.9)
df$residual.sugar[df$residual.sugar > hv] <- hv
boxplot(df$residual.sugar)

hv<- quantile(df$chlorides,0.90)
df$chlorides[df$chlorides > hv] <- hv
lv<-quantile(df$chlorides,0.10)
df$chlorides[df$chlorides < lv] <- lv
boxplot(df$chlorides)
summary(df$chlorides)

hv<- quantile(df$free.sulfur.dioxide,0.97)
df$free.sulfur.dioxide[df$free.sulfur.dioxide > hv] <- hv
boxplot(df$free.sulfur.dioxide)

hv<- quantile(df$total.sulfur.dioxide,0.95)
df$total.sulfur.dioxide[df$total.sulfur.dioxide > hv] <- hv
boxplot(df$total.sulfur.dioxide)

hv<- quantile(df$density,0.98)
df$density[df$density > hv] <- hv
lv<-quantile(df$density,0.02)
df$density[df$density < lv] <- lv
boxplot(df$density)

hv<- quantile(df$pH,0.98)
df$pH[df$pH > hv] <- hv
lv<-quantile(df$pH,0.02)
df$pH[df$pH < lv] <- lv
boxplot(df$pH)

hv<- quantile(df$sulphates,0.95)
df$sulphates[df$sulphates > hv] <- hv
boxplot(df$sulphates)

hv<- quantile(df$alcohol,0.99)
df$alcohol[df$alcohol > hv] <- hv
boxplot(df$alcohol)

boxplot(df[,1:11])
df$quality <-as.factor(df$quality)

summary(df)
summary(df$quality)

barplot(table(df$quality),col = c("magenta","yellow","red","green","pink","blue","orange"),xlab = "Quality",border = FALSE)

c<-ggplot(data = df,aes(x=quality))
c+geom_bar(aes(fill=quality))+coord_polar()


#TEST-TRAIN SPLIT

split <- sample.split(df,SplitRatio = 0.8)

Train_set = subset(df,split==TRUE)
Test_set = subset(df,split==FALSE)
View(Train_set)

#From the pairplot it can be figured out that linear types of models won't be able to classify the output variable properly
#due to non-linear boundary of separation.. Hence, it is wise to use KNN,Decision Trees and
#Radial adn polynomial Support Vector Machines.
#We would train the model on each kind of above model and choose the best model on the basis of the accucary.


# KNN MODEL

train_x = Train_set[,-12]
train_y = Train_set$quality

test_x = Test_set[,-12]
test_y= Test_set$quality



train_x_scale =scale(train_x)
test_x_scale = scale(test_x)

set.seed(0)
knn_pred = knn(train_x_scale,test_x_scale,train_y,k=1)

table(knn_pred,test_y)
#accuracy = 0.61



#Decision Tree

class_tree <- rpart(quality~.,data = Train_set,method = "class",control = rpart.control(maxdepth = 6))
rpart.plot(class_tree)

full_tree <- rpart(quality~.,data=Train_set,method = "class",control = rpart.control(cp=0))
rpart.plot(full_tree)

plotcp(class_tree)

class_tree$cptable
min_cp <- class_tree$cptable[which.min(class_tree$cptable[,"xerror"])]
plot(min_cp)

prun_tree <- prune(full_tree,cp=min_cp)
rpart.plot(prun_tree)


full_tree_pred =predict(full_tree,Test_set,type = "class")
table(full_tree_pred,Test_set$quality)
#accuracy = 0.575

prun_tree_pred =predict(prun_tree,Test_set,type = "class")
table(prun_tree_pred,Test_set$quality)
#accuracy = 0.57

class_tree_pred = predict(class_tree,Test_set,type = "class")
table(class_tree_pred,Test_set$quality)
#accuracy = 0.57


#SVM

svmfit <- svm(quality~.,data=Train_set,kernel="radial",gamma=1,cost=1)
summary(svmfit)


tune.out <- tune(svm,quality~.,data = Train_set,kernel="radial",ranges=list(cost=c(0.001,0.01,0.1,1,10,100,1000),gamma=c(0.01,0.5,0.1,1,5,10,100),cross=4))
summary(tune.out)


bestmodR = tune.out$best.model                                                                            

summary(bestmodR)

test_pred = predict(bestmodR,Test_set)

plot(test_pred)

table(test_pred,Test_set$quality)
#accuracy = 0.68 

Test_set_2 <- Test_set
Test_set_2$pred <- test_pred
ggplot(data=Test_set_2,aes(pred))+geom_bar(stat='count',aes(fill=quality))


tune.out.pol <- tune(svm,quality~.,data = Train_set,kernel="polynomial",ranges=list(cost=c(0.001,0.01,0.1,1,10,100,1000),degree=c(0.5,1,2,3,4,5),cross=4))
summary(tune.out.pol)

bestmodP=tune.out.pol$best.model
summary(bestmodP)

test_pred_p = predict(bestmodP,Test_set)

table(test_pred_p,Test_set$quality)
#accuracy = 0.6175

Test_set_1 <- Test_set
Test_set_1$pred <- test_pred_p
ggplot(data=Test_set_1,aes(pred))+geom_bar(stat='count',aes(fill=quality))


# Clearly, Supoort vector Machine -Radial is better suited model for above data set with the utmost accuracy.
               