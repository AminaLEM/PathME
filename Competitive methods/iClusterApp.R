library(Rdimtools)
library(iCluster)

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


data = list(as.matrix(x1),as.matrix(x2),as.matrix(x3))

# choose the best lambda and K based on the best POD
lamda = rlunif(50, 0.001, 0.9)
POD = matrix(0,8,50)
for (i in 1:50)
{
  for (j in 2:9)
{
fit = iCluster(data,j, lambda=c(lamda[i],lamda[i],lamda[i]))
POD[j-1,i]= compute.pod(fit)
}}

K= c(2,3,4,5,6,7,8,9)

matplot(K,POD, type = c("b"),pch=1,col = 1:50, xlab="Number of clusters") 
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

#Get iCluster clusters according to the chosen K and lambda, here k = 3 and lambda= 0.48
fit = iCluster(data,3, lambda=c(0.48,0.48,0.48))
clusters= fit$clusters
