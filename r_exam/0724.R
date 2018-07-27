####cluster####
install.packages("flexclust")
library(flexclust)

data(nutrient,package="flexclust")
row.names(nutrient) <- tolower(row.names(nutrient))
nutrient.scaled <- scale(nutrient)
d <- dist(nutrient.scaled)
fit.average <- hclust(d,method="average")
plot(fit.average,hang=1,cex=0.8)

install.packages("NbClust")
library(NbClust)
devAskNewPage(ask=TRUE)
nc <- NbClust(nutrient.scaled,distance="euclidean",min.nc=2,max.nc = 15,method="average")
barplot(table(nc$Best.nc[1,]))
clusters <- cutree(fit.average,k=5)
table(clusters)
aggregate(nutrient,by=list(cluster=clusters),median)
aggregate(as.data.frame(nutrient.scaled),by=list(cluster=clusters),median)
plot(fit.average,hang=-1,cex=0.8)
rect.hclust(fit.average,k=5)

floor(823.123,-1)

