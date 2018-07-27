name = c(
  "A",
  "A",
  "A",
  "A",
  "A",
  "A",
  "A",
  "B",
  "B",
  "B",
  "B",
  "B",
  "B",
  "B",
  "B",
  "C",
  "C",
  "C",
  "C",
  "C",
  "C",
  "C",
  "C",
  "C"
)
con = c(1, 4, 3, 3, 3, 3, 3, 4, 4, 3, 4, 4, 5, 4, 4, 4, 3, 4, 3, 4, 4, 3, 3, 3)
yA <- c(1, 4, 3, 3, 3, 3, 3)
yB <- c(4, 4, 3, 4, 4, 5, 4, 4)
yC <- c(4, 3, 4, 3, 4, 4, 3, 3, 3)
meanA <- mean(yA)
meanB <- mean(yB)
meanC <- mean(yC)
meantotal <- mean(con)
lA <- length(yA)
lA
lB <- length(yB)
lB
lC <- length(yC)
lC
lT <- length(con)
lT
r.df <- 3
e.df <- lT

ssr <-
  lA * sum((meanA - meantotal) ^ 2) + lB * sum((meanB - meantotal) ^ 2) + lC *
  sum((meanC - meantotal) ^ 2)
ssr
sse <-
  sum((yA - meanA) ^ 2) + sum((yB - meanB) ^ 2) + sum((yC - meanC) ^ 2)
sse
e.df
msr <- ssr / (r.df - 1)
msr
mse <- sse / (e.df - r.df)
mse
f <- msr / mse
f

1 - pf(f, (r.df - 1), (e.df - r.df))

name = factor(name)
data = data.frame(name, con)
View(data)
aov.result <- aov(con ~ name)
summary(aov.result)

y <-
  c(
    681,
    728,
    917,
    898,
    620,
    643,
    655,
    742,
    514,
    525,
    469,
    727,
    525,
    454,
    459,
    384,
    656,
    602,
    687,
    360
  )
x <- rep(c("소형", "준중형", "중형", "대형"), c(5, 5, 5, 5))
x <- factor(x)
y
x
aov.result <- aov(y ~ x)
summary(aov.result) #0.05보다 작으니 차종별로 차이가 있다.

#각각의 변수끼리 차이를 알 수 있음
TukeyHSD(aov.result)
par(las = 2)
par(mar = c(5, 8, 4, 2))
plot(TukeyHSD(aov.result))

#박스플랏으로
library(multcomp)
par(mar = c(5, 4, 6, 2))
tuk = glht(aov.result, linfct = mcp(trt = "Tukey"))

library(gplots)
par(mfrow = c(3, 3))

str(mtcars)
View(mtcars)
mtcars$cyl <- factor(mtcars$cyl)
aovm <- aov(mpg ~ cyl, data = mtcars)
summary(aovm)
plotmeans(mtcars$mpg ~ mtcars$cyl)


mtcars$am <- factor(mtcars$am)
aovm1 <- aov(mpg ~ am, data = mtcars)
summary(aovm1)
plotmeans(mtcars$mpg ~ mtcars$am)

mtcars$gear <- factor(mtcars$gear)
aovm2 <- aov(mpg ~ gear, data = mtcars)
summary(aovm2)
plotmeans(mtcars$mpg ~ mtcars$gear)

mtcars$vs <- factor(mtcars$vs)
aovm3 <- aov(mpg ~ vs, data = mtcars)
summary(aovm3)
plotmeans(mtcars$mpg ~ mtcars$vs)

mtcars$carb <- factor(mtcars$carb)
aovm4 <- aov(mpg ~ carb, data = mtcars)
summary(aovm4)
plotmeans(mtcars$mpg ~ mtcars$carb)

TukeyHSD(aovm)
par(las = 2)
par(mar = c(5, 8, 4, 2))
plot(TukeyHSD(aovm))

TukeyHSD(aovm1)
par(las = 2)
par(mar = c(5, 8, 4, 2))
plot(TukeyHSD(aovm1))

TukeyHSD(aovm2)
par(las = 2)
par(mar = c(5, 8, 4, 2))
plot(TukeyHSD(aovm2))

TukeyHSD(aovm3)
par(las = 2)
par(mar = c(5, 8, 4, 2))
plot(TukeyHSD(aovm3))

TukeyHSD(aovm4)
par(las = 2)
par(mar = c(5, 8, 4, 2))
plot(TukeyHSD(aovm4))

mtcars
mtcars$drat <- round(mtcars$drat)
mtcars$qsec <- round(mtcars$qsec)
mtcars$wt <- round(mtcars$wt)
mtcars$qsec <- round(mtcars$qsec, digits = 1)
str(mtcars)


mtcars$qsec <- factor(mtcars$qsec)
aovm5 <- aov(mpg ~ qsec, data = mtcars)
summary(aovm5)
plotmeans(mtcars$mpg ~ mtcars$qsec)

mtcars$wt <- factor(mtcars$wt)
aovm6 <- aov(mpg ~ wt, data = mtcars)
summary(aovm6)
plotmeans(mtcars$mpg ~ mtcars$wt)

str(ToothGrowth)
attach(ToothGrowth)
table(supp, dose)
aggregate(len, by = list(supp, dose), FUN = mean)
aggregate(len, by = list(supp, dose), FUN = sd)

fit <- aov(len ~ supp * dose)
summary(fit)


taov <- aov(mpg ~ cyl + gear, data = mtcars)
summary(taov)

taov <- aov(mpg ~ gear + cyl, data = mtcars)
summary(taov)

taov <- aov(mpg ~ cyl * gear, data = mtcars)
summary(taov)

taov <- aov(mpg ~ gear * cyl, data = mtcars)
summary(taov)

library(HH)
interaction2wt(mpg ~ gear * cyl)
interaction2wt(mpg ~ cyl * gear)

aggregate(mpg, by = list(cyl), FUN = mean)
? aggregate
