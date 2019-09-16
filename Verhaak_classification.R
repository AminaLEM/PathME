verhaak= read.table('D:/newflder/TCGA_unified_CORE_ClaNC840.txt', sep ='\t', header = F) # genes*samples
rownames(verhaak) = verhaak[,1]
verhaak = verhaak[,c(-1,-2)]
gene840_names = rownames(verhaak)
samples= t(verhaak[1,])
sample_verhaak =  gsub("-","\\.",samples[order(samples)])

# extract the three first parts of TCGA samples from verhaak samples
sample_verhaak = substr(sample_verhaak,1,12)
classes = t(verhaak[2,])
classes = classes[order(samples) ]
classesIN = t(verhaak[1:2,])
classesIN[,1]= substr(gsub("-","\\.",classesIN[,1]),1,12)

#verhaak gene expression matrix 
vnh2= read.table('D:/newflder/unifiedScaled.txt', sep= '\t',header=1)
rows_indx = which(rownames(vnh2) %in% gene840_names)
col_indx = which(substr(colnames(vnh2),1,12) %in% sample_verhaak)
vrhCoreGE= vnh2[rows_indx,col_indx]

# our data 
mydata = t(mydatGE) # genes * samples

#select only rows associated to genes considered in verhaak classification (842 genes)
genes_mydata = rownames(mydata)
mymatrix = mydata[which(genes_mydata %in% gene840_names),]

# select samples from mymatrix that are not considered in verhaak classification
target_mymatrix = mymatrix[,which(!colnames(mymatrix) %in% sample_verhaak)]

## Replace the class names ["Proneural","Neural","Classical","Mesenchymal"] to be the numbers in 1:4.
class_names = 1:4
id = as.integer(factor(classes))
## use cross-validation to estimate the error rates for
## classifiers of different sizes (different numbers of genes) 
data= as.matrix(vrhCoreGE)
class(data) <- "numeric" 
#cv_out = cvClanc(data, id,active = 1:200)

## View the estimated error rates associated with different feature-set sizes.
#plot(1:200, cv_out$overallErrors, type = "l", lwd = 2, col = "blue", xlab = 
#      "Number of features", ylab = "CV error")

# build the verhaak classifier

train_out = trainClanc(data , id, gene840_names)
build_out = buildClanc(data, id, class_names, train_out, active = 840)

# Predict classes for our samples
classes_mysamples= predictClanc(data=scale(target_mymatrix), geneNames=gene840_names, fit=build_out)
verhaakclasses= matrix(NA,273,2)

for (i in 1:273)
{  verhaakclasses[i,1] = colnames(mymatrix)[i]
  if(colnames(mymatrix)[i] %in% sample_verhaak)
  verhaakclasses[i,2] = classesIN[which(classesIN[,1] %in% colnames(mymatrix)[i]),2]
  
 else
    verhaakclasses[i,2]= classes_mysamples[which(colnames(target_mymatrix) %in% colnames(mymatrix)[i])]
}
