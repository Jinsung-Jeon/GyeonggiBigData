cran <- getOption("repos")
cran["dmlc"] <-
  "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
install.packages("stringi", dependencies = T)
help(package = 'mxnet')
library(mxnet)

data(Sonar, package = "mlbench")
#mxnet은 수치형만을 요구 M,R -> 1,2 -> 0,1
Sonar[, 61] <- as.numeric(Sonar[, 61]) - 1

#dummy 사용
#contrasts(iris$Species,contrasts = T)
#contrasts(iris$Species,contrasts = F)

#mxnet은 matrix형태만 요구
train.ind <- c(1:50, 100:150)
train.x <- data.matrix(Sonar[train.ind, 1:60])
train.y <- Sonar[train.ind, 61]
test.x <- data.matrix(Sonar[-train.ind, 1:60])
test.y <- Sonar[-train.ind, 61]
mx.set.seed(0)
#?mx.mlp hidden_node vector로 들어가기때문에 hidden layer 결정가능
# output이 0,1로만 나오게하려면 sigmoid, output이 a,b 01,10으로 나오게하려면 softmax
model <-
  mx.mlp(
    train.x,
    train.y,
    hidden_node = 10,
    out_node = 2,
    out_activation = 'softmax',
    num.round = 20,
    array.batch.size = 15,
    learning.rate = 0.07,
    momentum = 0.9,
    eval.metric = mx.metric.accuracy
  )

preds <- predict(model, test.x)
pred.label <- max.col(t(preds)) - 1
table(pred.label, test.y)


#### DNN ####

#data preparing
train <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_train.csv')
test <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_test.csv')

train <- data.matrix(train)
test <- data.matrix(test)
train.x <- train[,-1]
train.y <- train[,1]
train.x <- t(train.x/255) #표준화
test_org <- test #원본파일 저장
test <- test[,-1]
test <- t(test/255)
table(train.y)

#data analysis
data <- mx.symbol.Variable("data") #input을 data로 선언
fc1 <-
  mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 128) #은닉층
act1 <-
  mx.symbol.Activation(fc1, name = "relu1", act_type = "relu") #활성함수
fc2 <-
  mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64) #은닉층
act2 <-
  mx.symbol.Activation(fc2, name = "relu2", act_type = "relu") #활성함수
fc3 <-
  mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = 10) #은닉층
softmax <- mx.symbol.SoftmaxOutput(fc3, name = "sm") #마지막활성함수
devices <- mx.cpu() #cpu를 사용함
mx.set.seed(0)
#?mx.model.FeedForward.create
model <-
  mx.model.FeedForward.create(
    softmax,
    X = train.x,
    y = train.y,
    ctx = devices,#cpu에서 돌린다
    num.round = 10,
    array.batch.size = 100,
    learning.rate  = 0.07,
    momentum = 0.9,
    eval.metric = mx.metric.accuracy,
    initializer = mx.init.uniform(0.07), #초기값 제공
    epoch.end.callback = mx.callback.log.train.metric(100)
  )

pred <- predict(model, test)
pred
dim(pred)
pred.label <- max.col(t(pred)) -1
table(pred.label)
head(pred.label)
t1 <- table(test_org[,1],pred.label)
sum(diag(t1))/sum(t1)

#### 마이닝 sample ####
#https://github.com/ozt-ca


#### tensorflow ####
#https://github.com/rstudio/tensorflow
#https://tensorflowkorea.gitbooks.io/tensorflow-kr/ 한글 설명