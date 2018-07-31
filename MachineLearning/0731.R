library(caret)
library(mlbench)
library(e1071)

data(Vowel)
data(Sonar)

set.seed(100)
ind1 <- createDataPartition(Sonar$Class, p=0.7)
ind2 <- createDataPartition(Vowel$Class, p=0.7)

train1 <- Sonar[ind1$Resample1,]
test1 <- Sonar[-ind1$Resample1,]
train2 <- Vowel[ind2$Resample1,-1]
test2 <- Vowel[-ind2$Resample1,-1]

m1 <- svm(Class~., train1)
m1$nSV #support vector 각각 개수
m1$tot.nSV #support vector 총합

m1 <- svm(Class~., train1, cost=100)
m1$nSV #support vector 각각 개수
m1$tot.nSV #support vector 총합

m1 <- svm(Class~., train1, cost=300)
m1$nSV #support vector 각각 개수
m1$tot.nSV #support vector 총합

m1 <- svm(Class~., train1, cost=0.1)
m1$nSV #support vector 각각 개수
m1$tot.nSV #support vector 총합

m1$labels
m1$SV #scale 해준 값 
m1$coefs #y=wx+b 에서의 w 값 

pred1 <- predict(m1,test1)
confusionMatrix(pred1, test1$Class)$overall[1]

m2 <- svm(Class~., train2)
pred2 <- predict(m2,test2)
confusionMatrix(pred2, test2$Class)$overall[1]

data1 <- read.table("data1.txt", header=F, sep = ',',encoding = 'UTF-8')
names(data1) <- c('X','Y')
plot(data1)
m3 <- lm(Y~X, data1)
lines(data1$X, m3$fitted.values,lty=3, col='blue')
m4 <- svm(Y~X, data1)
lines(data1$X,m4$fitted, lty=2, col='red')
m4$kernel
m2$kernel
?kernel

m5 <- svm(Y~X, data1, kernel='radial')
lines(data1$X, m5$fitted,lty=3, col='green')

#tune 10번, epsillon 11번  cost 8번 총 10x11x8번 돌림
tuneResult <- tune(svm, Y ~ X,  data = data1, ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9))
)
tuneResult
lines(data1$X,tuneResult$best.model$fitted, col="green") #best

#### img data ####

mnist_tr01 <- read.csv(choose.files(),header=F)
mnist_te01 <- read.csv(choose.files(),header=F)

library(caret)
names(mnist_tr01) <- c('Y', paste('V',1:(28*28),sep=''))
names(mnist_te01) <- c('Y', paste('V',1:(28*28),sep=''))
names(mnist_tr01)
str(mnist_tr01)
mnist_tr01$Y <- as.factor(mnist_tr01$Y)
mnist_te01$Y <- as.factor(mnist_te01$Y)

train3 <- createDataPartition(mnist_tr01$Y, p=0.1)
test3 <- createDataPartition(mnist_te01$Y, p=0.1)

train4 <- mnist_tr01[train3$Resample1,]
test4 <- mnist_te01[train3$Resample1,]


mat1 <- as.matrix(train4[1,-1],nrow=28)
str(mat1)
mat2 <- mat1/255
mat3 <- matrix(mat2,nrow=28)
image(mat3)
train4$Y[1]

str(train4)
dim(train4)
dim(mnist_tr01)

m5 <- svm(Y~.,train4)
pred5 <- predict(m5, test4)
c3 <- confusionMatrix(pred5, test4$Y)

train5 <- train4
train5[,-1] <- sapply(train5[,-1],function(x) x/255)

m6 <- svm(Y~., train5)

source("https://bioconductor.org/biocLite.R")
biocLite("EBImage")

library("EBImage")
install.packages("readbitmap")
library(jpeg)
str(img001)

img1 <- resize(readJPEG('img01.jpg'),50,50)
class(img1)
dim(img1)
r1=50
c1=50
ch1=3
matrix(ncol=r1*c1*ch1)
dir2 <- list.dirs('.')
for( i in 1:length(dir2)){
  files2 <- list.files(dir2[i])
  for (j in 1:length(files2))
  {
    img1 <- resize(readJPEG(files2[j]),r1,c1)
    mat1 <- rbind(mat1, img1)
    }
}
dir1
files2 <- list.files(dir1[8])
files2
m6 <- svm(Y~., train5)
m6$coefs

Mxnet