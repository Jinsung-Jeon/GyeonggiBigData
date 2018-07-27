#R에서 SQL 불러오기 

install.packages("RMySQL")
library(RMySQL)

con =dbConnect(drv = MySQL(),
          dbname = "sakila",
          user = "root",
          password= "1234",
          host= "localhost",
          port = 3306)

#install.packages("rvest") #원래는 크롤링할때 사용함. 인코딩해줌 
##library(rvest)
##repair_encoding(dbListTables(conn = con))
dbListTables(conn = con) #db연결

dbListFields(con, "film")

#paste("select * from", tables[1])
sakila_film=dbGetQuery(con, "select * from film")
str(sakila_film)

dbDisconnect(con) #db종료


## 기술 통계학 ## 
install.packages("vcd")
library(vcd)

counts = table(Arthritis$Improved)
barplot(counts,
        main = "simple Bar plot",
        xlab = "improvement",
        ylab = "Frequency")

#그림파일로 저장
png(filename = "simple bar plot.png",
    height = 400,
    width = 400)
barplot(counts,
        main = "simple Bar plot",
        xlab = "improvement",
        ylab = "Frequency")
dev.off()

plot(Arthritis$Improved,
     main = "simple Bar plot",
     xlab = "improvement",
     ylab = "Frequency" )
methods("plot")

counts1 = table(Arthritis$Improved, Arthritis$Treatment)

barplot(counts1,xlab = "Treatment",ylab = "Frequency",col = c("red","green","whitesmoke"),legend = rownames(counts1),beside = T)
#beside = T 누적차트
spine(counts1) #bar 높이를 1로 

install.packages("plotrix")
library(plotrix)
par(mfrow =c(1,1)) #표여러개만들자
pie(counts)
pie3D(counts, labels = rownames(counts))

#연속형 히스토그램
mtcars$mpg
hist01=hist(mtcars$mpg, breaks = 12, freq = F) #freq=F -> 전체넓이=1 , freq=T -> 갯수
lines(density(mtcars$mpg),
      col = "blue",
      lwd = 2)
dev.off

hist01$counts

install.packages("sm")
library(sm)

mtcars$cyl
mtcars$mpg

dev.off

cy1.f = factor(mtcars$cyl,levels = c(4,6,8),labels = c("4 cylinder","6 cylinder","8 cylinder"))
sm.density.compare(mtcars$mpg, mtcars$cyl, xlab = "Miles per Gallon")
title(main = "MPG.")
colfill = c(2:(1+length(levels(cy1.f))))
legend(locator(1), levels(cy1.f),fill=colfill)

stat=boxplot(mtcars$mpg, main = "Box Plot", ylab = "Miles per Gallon")
stat$out #이상치유무

boxplot(mpg ~ cyl, data = mtcars)

dotchart(mtcars$mpg, labels = rownames(mtcars),cex = 1)

myvars = c("mpg","hp","wt")
summary(mtcars[myvars])

View(mtcars)
var(mtcars$mpg)
sd(mtcars$mpg)
boxplot(mtcars$mpg)

sd(mtcars$hp)
boxplot(mtcars$hp)

sd(mtcars$disp)
boxplot(mtcars$disp)
hist(mtcars$disp)