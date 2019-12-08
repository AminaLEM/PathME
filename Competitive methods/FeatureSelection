\\ features selection based on the variance (sorting features in decreasing order by variance and then picking the top k ones, k may be diffrent across datasets)
k = 200 // Number of features to select
\\ Gene expression dataset
dat= inputge
matr= rep(0,dim(dat)[2])
for (i in 1:dim(dat)[2])
{
  matr[i]= var(dat[,i])
  
}
ind= tail(order(matr),k)
X1= dat[,ind]

\\ CNV dataset
dat= inputcnv
matr= rep(0,dim(dat)[2])
for (i in 1:dim(dat)[2])
{
  matr[i]= var(dat[,i])
  
}
ind= tail(order(matr),k)
X2= dat[,ind]

\\ miRNA dataset
dat= inputmiRNA
matr= rep(0,dim(dat)[2])
for (i in 1:dim(dat)[2])
{ 
  matr[i]= var(dat[,i])
  
}

ind= tail(order(matr),k)
X3= dat[,ind]



