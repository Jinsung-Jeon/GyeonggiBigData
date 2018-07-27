library1 <- c("plyr","ggplot2","stringr","zoo","corrplot","RColorBrewer")
unlist(lapply(library1,require,character.only=TRUE))

product <- read.csv("C:\\Users\\callab\\Dropbox\\r_exam\\농산물\\Data\\product.csv",header=T,fileEncoding="CP949")
code <- read.csv("C:\\Users\\callab\\Dropbox\\r_exam\\농산물\\Data\\code.csv",header=T,fileEncoding="CP949")

colnames(product) <- c('date','category','item','region','mart','price')
category <- subset(code,code$구분코드설명=="품목코드")
colnames(category) <- c('code','exp','item','name')
total.pig <- product[which(product$item==514),]
head(total.pig)
region <- subset(code,code$구분코드설명=="지역코드")
colnames(region) <- c('code','exp','region','name')
day.pig <- merge(total.pig,region,by="region",all=T)
head(day.pig,n=10)
total.pig.mean <- dlply(ddply(ddply(day.pig,.(date),summarise,name=name,region=region,price=price),.(date,name),summarise,mean.price=mean(price)),.(name))
x <- data.frame(Date=as.Date(c('2013-10-01','2013-10-02','2013-10-02','2013-10-02','2013-10-01','2013-10-02','2013-10-02')),
                Category=factor(c('First','First','First','Second','Third','Third','Second')),frequency=c(10,15,5,2,14,20,3))
ddply(x,.(Date,Category),summarize,Sum_F=sum(frequency))
dlply(x,.(Date),summarize,Sum_F=sum(frequency))
for(i in 1: length(total.pig.mean)){
  cat(names(total.pig.mean)[i],"의 데이터의 길이는",nrow(total.pig.mean[[i]]),"이다","\n")
}

day.pig <- day.pig[! day.pig$name %in% c("의정부","용인","창원","안동","포항","순천","춘천"),]
pig.region.daily.mean <- ddply(day.pig,.(name,region,date),summarise,mean.price=mean(price))
head(pig.region.daily.mean, n=10)
head(x)
