install.packages("VIM")
library(VIM)

View(sleep)
aggr(sleep, prop=FALSE)
aggr(sp,prop=FALSE)
sleep
marginplot(sleep[c("Gest","Dream")],pch=c(20), col=c("darkgray","red","blue"))
marginplot(sp[c("target","Date")],pch=c(20), col=c("darkgray","red","blue"))
sp$Date = factor(sp$Date)
str(sp)
matrixplot(sp)

x <-  as.data.frame(abs(is.na(sleep)))
head(x)
y <- x[which(apply(x,2,sum)>0)]
y
cor(y)

x1 <-  as.data.frame(abs(is.na(sp)))
head(x1)
y1 <- x1[which(apply(x1,2,sum)>0)]
y1

cor(y1)
