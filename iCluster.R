library(Rdimtools)
library(iCluster)
library( KScorrect)
inputge= mydatGE
inputcnv= mydatME
inputmiRNA= mydatMI

inputge = read.csv("G:/brcadata/RNASEQ.csv", sep = ",", header = T)
inputcnv = read.csv("G:/brcadata/CNV.csv", sep = ",", header = T)
inputmiRNA = read.csv("G:/brcadata/miRNASEQ.csv", sep = ",", header = T)
rownames(inputge)= inputge[,1]
inputge = inputge[,-1]

rownames(inputcnv)= inputcnv[,1]
inputcnv = inputcnv[,-1]


rownames(inputmiRNA)= inputmiRNA[,1]
inputmiRNA = inputmiRNA[,-1]


#X= as.matrix(mydatGE)
X= as.matrix(inputge)
x1 = do.lscore(X,100)
write.csv(x1$Y, './mydatge100.csv')

#X=as.matrix( mydatME)
X= as.matrix(inputcnv)
x2 = do.lscore(X, 100)
write.csv(x1$Y, './mydatme100.csv')

#X= as.matrix(mydatMI)
X= as.matrix(inputmiRNA)
x3 = do.lscore(X, 200)

write.csv(x1$Y, './mydatmi100.csv')

data = list(as.matrix(x1),as.matrix(x2),as.matrix(x3))

lamda = rlunif(50, 0.001, 0.9)


POD = matrix(0,8,50)


for (i in 1:5)
{
  for (j in 2:9)
{
fit = iCluster(data,j, lambda=c(lamda[i],lamda[i],lamda[i]))
POD[j-1,i]= compute.pod(fit)
}}

POD[1:5,4]=POD[1:5,2]
POD[2,26]=POD[2,27]

POD[1,24]=POD[1,25]
POD[2,24]=POD[2,25]

POD[1,41]=POD[1,42]
POD[2,41]=POD[2,42]
POD[3,41]=POD[3,42]

K= c(2,3,4,5,6,7,8,9)

matplot(K,POD, type = c("b"),pch=1,col = 1:50, xlab="Number of clusters") #plot
lam = order(lamda)
my.range <- range(t(POD[2,lam]))
my.range <-c(min(range(t(POD[4,lam]))[1],range(t(POD[1,lam]))[1]), 1.4*max(range(t(POD[4,lam]))[2],range(t(POD[1,lam]))[2]))

matplot(log(lamda[lam]),t(POD[1:3,lam]), lwd = 2,ylim = my.range, type = c("b"),pch =16,col = 1:3, ylab="POD",
     xlab= expression(log(lambda))
) #plot
lines(log(lamda[lam]),t(POD[3,lam]), lwd = 2, type = c("b"),pch =16, col=3)

lines(log(lamda[lam]),t(POD[4,lam]), lwd = 2, type = c("b"),pch =16, col=4)
lines(log(lamda[lam]),t(POD[5,lam]), lwd = 2, type = c("b"),pch =16, col=5)

lines(log(lamda[lam]),t(POD[6,lam]), lwd = 2, type = c("b"),pch =16, col="#A211E5")
lines(log(lamda[lam]),t(POD[7,lam]), lwd = 2, type = c("b"),pch =16,col="#ED1379")

lines(log(lamda[lam]),t(POD[8,lam]), lwd = 2, type = c("b"),pch =16, col="#D4EC3D")

legend("topleft",horiz=TRUE, inset=.02,  title="Number of clusters", legend = 2:9, col=c(1,2,3,4,5,"#A211E5","#ED1379","#D4EC3D"), lwd=2,cex=0.8) # optional legend



plot(log(lamda[lam]),POD[1,lam], type = c("b")
        ,pch=1,col = 4, ylab="POD",lwd=2,
     xlab= expression(log(lambda))
     ) #plot

lam = order(lamda)

fit = iCluster(data,3, lambda=c(0.48,0.48,0.48))
group= fit$clusters
compute.pod(fit)
survdiff(Surv(as.numeric(survival[,2]),as.numeric(survival[,3])) ~ group)

survdiff(Surv(as.numeric(clinc[,3]),as.numeric(clinc[,3])) ~ group)

clinc = survival
library(survival)
grp = read.csv("H:/Nouveau dossier (3)/gr2gbm100.csv", header=F)
group = grp[,1]
fitt=survfit(Surv(as.numeric(clinc[,2]),as.numeric(clinc[,3])) ~ clusters)

plot(fitt,col=c(1:4),xlab = "time",ylab = "survival probability")
legend (2200,1,c("subtype1","subtype2"), col=(1:4), lwd=0.5,cex=0.8)
title("SNF")


survdiff(Surv(as.numeric(clinc[,2]),as.numeric(clinc[,3])) ~ group)

library(cluster)
si = silhouette(clusters, dist(dat))
mean(si[,3])

len= dim(mydatMut)[2] 
mutation<- vector(mode="character", length=len)
cluster1=  vector(mode="double", length=len)
cluster2 =  vector(mode="double", length=len)
cluster3=  vector(mode="double", length=len)
cluster4=  vector(mode="double", length=len)

test = data.frame(mutation,cluster1,
                  cluster2)

test$mutation= ""

for(j in 1:len)
{
  test$mutation[j]= colnames(mydatMut)[j]
  successsample = length(which(mydatMut[which(group == 1),j]==1))
  successleftpart=  length(which(mydatMut[,j]==1))-successsample
  failuresamples=length(which(group == 1)) - successsample
  failureleftpart = length(which(mydatMut[,j]==0))-failuresamples
  test$cluster1[j] = fisher.test(matrix(c(successsample, successleftpart,failuresamples, failureleftpart), 2, 2), alternative='greater')$p.value; 
  #phyper(successsample-1, successbkgd, failurebkgd, samplesize, lower.tail=FALSE);
  
  
}
test$cluster1 = p.adjust(test$cluster1 , 'BH')


for(j in 1:len)
{
  successsample = length(which(mydatMut[which(group == 2),j]==1))
  successleftpart=  length(which(mydatMut[,j]==1))-successsample
  failuresamples=length(which(group == 2)) - successsample
  failureleftpart = length(which(mydatMut[,j]==0))-failuresamples
  test$cluster2[j] = fisher.test(matrix(c(successsample, successleftpart,failuresamples, failureleftpart), 2, 2), alternative='greater')$p.value; 
  #phyper(successsample-1, successbkgd, failurebkgd, samplesize, lower.tail=FALSE);
  
  
}
test$cluster2 = p.adjust(test$cluster2 , 'BH')
bar4 <- subset(test, mutation %in% test$mutation[which(test$cluster1 <0.05)] | mutation %in% test$mutation[which(test$cluster2 <0.05)] )
dim(bar4)



compute.pod(fit)
