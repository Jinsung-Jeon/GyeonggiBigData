car = cars
?cars
str(cars)

#2개의 변수가 수치형으로 되어있으면 50개의 sample이 dataframe 형태로 들어있다.

x = c(5,7,8,9)
y = c(4,8,6,10)
z = c("A","B","C","D")
x[2] == y[3]
x%/%y
matrix(x,nrow=2)
data = data.frame(x,y,z)

salary= read.csv("salary.csv")
str(salary)
# 6개의 변수로 이루어져있으며 X와 year는 정수형  negotiated 논리형 gender는 팩터
# 나머지 변수는 수치형으로 이뤄져있다.

install.packages("ggplot2")
library(ggplot2)
str(midwest)

names(midwest)[5] = 'total'
names(midwest)[10] = 'asian'

midwest$per_asian=(midwest$asian/midwest$total)*100
plot(midwest$per_asian)
average_asian=(sum(midwest$asian)/sum(midwest$total))*100
midwest$lspop
midwest$lspop[midwest$per_asian > average_asian] ='large'
midwest$lspop[midwest$per_asian < average_asian] ='small'
table(midwest$lspop)


#### 흐름제어와 함수 ####
ifelse{
  test, #참, 거짓을 저장한 객체
  yes, #test가 참일 때 선택할 값
  no #test가 거짓일 때 선택할 값 
}

x = c(1,2,3,4,5)
ifelse(x%%2 ==0,"even","odd")

x = c(1,2,3,4,5)
switch(x[2],"1"=print("one"),"2"=print("two"),"3"=print("three"),print("NOT"))

for(i in 1:10){
  print(i)
}

i = 1
while(i <= 10){
  i = i
  print(i)
}

#피보나치
fibo = function(n){
  if(n==1||n==2){
    return(1)
  }
  return(fibo(n-1)+fibo(n-2))
}

#원넓이
round = function(r){
  return(r^2*pi)
}
round(4)

#사다리꼴넓이
rec = function(u,l,h){
  return(((u+l)/2)*h)
}
rec(3,4,5)
