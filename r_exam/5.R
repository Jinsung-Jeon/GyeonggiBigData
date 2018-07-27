str(cars)
plot(cars) #상관관계있음
boxplot(cars$speed,cars$dist) #이상치 있음

str(mtcars)
summary(mtcars) #수치가 다름능 -> 표준화 필요함
boxplot(mtcars) #이상치 있음
cor_car=cor(mtcars)
par(mfrow=c(1,2))
install.packages("corrgram")
library(corrgram)
corrgram(cor_car, upper.panel = panel.conf) #변수간 상관관계 확인가능
View(mtcars)


ti_test=read.csv("test.csv")
ti_train=read.csv("train.csv")
ti_train = ti_train[,-2]
titanic=rbind(ti_train, ti_test)
str(titanic)
summary(titanic) # 고유값
table(is.na(titanic)) #NA 존재
hist(table(titanic$Pclass))
hist(titanic$Pclass)
table(titanic$Age) #이상값 존재
summary(titanic$Embarked)
num_ti=summary(titanic$Ticket) #같이 탄 사람의 수가 11명이 제일 많음
summary(titanic$Fare) 
boxplot(titanic$Fare)
plot(titanic$Fare,titanic$)
View(num_ti)



a = c(13.60,15.15,17.62,16.81,15.51,15.12,14.39,15.20,13.70,14.75,15.13,15.66,13.69,15.74,14.96,15.20,16.45,13.66,16.16,14.47)
b = c(13.77,13.63,12.63,14.13,13.50,13.09,13.96,13.41,14.03,14.25,13.47,13.43,13.24,14.61,13.82,14.07,15.96,13.69,14.25,14.50)
box = boxplot(a,b,names=c("a","b"))
box$stats
box$out
var.test(a,b)
a.correct = a[a!=box$out[1]]
b.correct = b[b!=box$out[2]]
shapiro.test(a.correct) #정규분포를 따른다.

#두집단의 정규분포 테스트
mean_z.test_two <- function(x,y,sigma1,sigma2, conf.level,alternative){
box = boxplot(a.correct,b.correct,names=c("a","b"))
n=length(a)
m=length(b)
mean_x=mean(a) 
mean_y=mean(b)
diff = mean_x-mean_y
z_alpha_half <- qnorm((1-conf.level)/2,lower.tail = FALSE) 
var_mean <- (sigma1^2/n)+(sigma2^2/m)
lower_limit <- diff-z_alpha_half*sqrt(var_mean)
upper_limit <- diff+z_alpha_half*sqrt(var_mean)
z_statistic <- diff/sqrt(var_mean)
p_value_R <- pnorm(z_statistic, lower.tail = FALSE)
p_value_L <- pnorm(z_statistic, lower.tail = TRUE) 
p_value_two <- pnorm(abs(z_statistic), lower.tail = FALSE)
if(alternative == "two_sided"){ 
  p_value <- 2*p_value_two 
  cat("Two sample Z-test", "\n")
}else if(alternative == "greater"){ 
  p_value <- p_value_R 
  cat("Two sample Z-test", "\n") 
}else { 
  p_value <- p_value_L 
  cat("Two sample Z-test", "\n") 
}
result <- ifelse(p_value < (1-conf.level), "Reject H0", "Accept H0")
cat("Z_statistics = ", z_statistic, "\n")
cat("p_value = ", p_value, "\n")
if(alternative == "two_sided"){ 
  cat("alternative hyphothesis: true difference in mean is not equal to 0 ", "\n") 
}else if(alternative == "greater"){ 
  cat("alternative hyphothesis: true difference in mean is greater than 0 ", "\n") 
}else{ 
  cat("alternative hyphothesis: true difference in mean is less than 0 ", "\n") 
} 
cat(conf.level * 100, "% confidence interval :", lower_limit, upper_limit, "\n") 
cat("sample mean of x = ", mean_x, ", sample mean of y = ", mean_y, "\n") 
cat("sample estimate difference of mean=", mean_x - mean_y, "\n")
cat("result = ", result)}



a.v=var(a)
b.v=var(b)

(a.m-b.m)/sqrt((a.v/a.n)+(b.v/b.n))

a = c(91,115,96,90,120,108,82,118,105,97,113,119,90,106,116,92,108,115,114,101,96,96,96,97,89,99,90,85,91,124,93,90,100,100,91,96,120,78,96,114)
b = c(102,117,82,104,77,110,93,115,75,103,126,79,81,118,93,106,104,97,115,80,78,109,116,104,102,137,99,100,113,112,96,106,76,102,111,105,85,125,77,111)
var(a)
var(b)

boxplot(a,b) #이상치가 없고 b의 분산이 더 크ㄷ

a.m=mean(a)
a.d=sd(a)
a.n=length(a)

b.m=mean(b)
b.d=sd(b)
b.n=length(b)

shapiro.test(a) #정규분포 따르지않는다
shapiro.test(b) #정규분포 따른다.

a.b.mean= mean(a-b)
a.b.var=var(a-b)

a.b.mean/sqrt(a.b.var/39)

shapiro.test(a-b)
(a.m-b.m)-z_val*sqrt((a.d^2)/a.n+(b.d^2)/b.n)
sigma1 = 100
sigma2 = 225
conf.level = 0.95
alternative = "two_sided"
mean_z.test_two(x=a, y=b, sigma1 = sigma1, sigma2=sigma2,conf.level=conf.level, alternative=alternative)




(a.m-b.m)-z_val*sqrt((a.d^2)/a.n+(b.d^2)/b.n)
(a.m-b.m)+z_val*sqrt((a.d^2)/a.n+(b.d^2)/b.n)
z_val=qnorm(1-0.025)
t.test(a,b,mu=0, paired=TRUE, alternative="two.sided")