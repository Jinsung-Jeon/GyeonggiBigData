##실습2

a = c(26,16,20,7,22,15,28,17,3,1,16,19,13,27,4,30,8,3,12)
b = c(1267,887,1022,511,1193,795,1477,991,455,324,944,1232,808,1296,486,1516,565,299,830)
pay = data.frame(a,b)
pay[pay$b] = sort(pay$b)
pay
m = lm(b~a, data = pay)
summary(m)
par(mfrow=c(2,2))
plot(m)
x_bar = seq(min(a),max(a),5)
length(x_bar)
str(x_bar)

p = predict(m,data.frame(a=x_bar), interval = "confidence")
p
matplot(x_bar,p, type='n') 
matlines(x_bar,p, lty=c(1, 2, 2), col=1)

dev.off()

##실습3
install.packages("UsingR")
library(UsingR)
data = galton

?sort
m = lm(child~parent, data = data)
summary(m)
par(mfrow=c(2,2))
plot(m)

shapiro.test(residuals(m)) #정규성이 없으면 비모수적 방법의 회귀분석을 해야함
x_bar = seq(min(data$child),max(data$child),1)
str(x_bar)
p = predict(m,newdata =data.frame(parent=x_bar), interval = "confidence")
p
?predict
matplot(x_bar,p, type='n') 
matlines(x_bar,p, lty=c(1, 2, 2), col=1)

View(women)
wolm = lm(weight~height,data=women)
shapiro.test(residuals(wolm))
plot(women$height,women$weight)
plot(wolm)
p


x1 <- c(507,391, 488, 223, 274, 287, 550, 457, 377, 101, 170, 450, 309, 291, 375, 198, 641, 528, 500, 570) 
x2 <-c("F","F","F","F","F","F","F","F","F","F", "M","M","M","M","M","M","M","M","M","M") 
y <- c(1096, 759, 965, 698, 765, 703, 968, 805, 912, 588, 281, 527, 439, 318, 412, 370, 1044, 537, 649, 807) 
data1 = data.frame(x1,x2,y)

tlm = lm(y~x1)
summary(tlm)

t2lm = lm(y~x1+x2)
summary(t2lm)

t3lm = lm(y~x1+x2+x1:x2)
summary(t3lm)

anova(tlm,t2lm,t3lm)

coef <- coefficients(t2lm)
coef
plot(x1, y, pch=x2)
abline(tlm)
abline(coef[1],coef[2],lty=2, col="red")
abline(coef[1]+coef[3],coef[2],lty=2, col="blue")
legend(locator(1),c("F","F+M","M"),lty=c(2,1,3),col=c("red","black","blue"))

str(mtcars)


library(MASS)
state.x77
states = as.data.frame(state.x77[,c(5,1,3,2,7)])
fit <-lm(Murder ~ . ,data=states)
stepAIC(fit,direction = "backward")

cars=mtcars
mt_lm = lm(mpg~.,data=mtcars)
summary(mt_lm)

stepAIC(mt_lm,direction = "backward")

install.packages("leaps")
library(leaps)
leaps = regsubsets(mpg~.,data = cars, nbest = 4)
plot(leaps, scale="adjr2")

m2 <- lm(mpg~ wt+qsec+am+carb,data=cars)
summary(m2)
vif(m2)

m3 <- lm(mpg~ hp+wt+qsec+am,data=cars)
summary(m3)
vif(m3)

m4 <- lm(mpg~ disp+hp+wt+qsec+am, data=cars)
summary(m4)
vif(m4)

anova(m2,m3,m4)

subsets(leaps,statistic = "cp")
abline(1,1,lty=2,col="red")

mt.cor=cor(mtcars[,-1])
install.packages("prcomp")
library(prcomp)
mt.pr=prcomp(mt.cor, scale=TRUE)
summary(mt.pr)
View(mtcars)

leaps = regsubsets(mpg~.,data = cars, nbest = 2)
plot(leaps, scale="adjr2")
