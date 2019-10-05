library(NMF)
library(Survival)
beta = 0.01
nrun =30
#load our resulted matrix of [patients X pathways] and clinical data
load(data_matrix)
load(survival_data)
load(clinical_data)
load(pathway_names)

# Scale data in the interval [0,1] 
data_matrix = data_matrix +  abs(min(data_matrix))
data_matrix = data_matrix/max(data_matrix)

# Estimate the rank of factorization by boxplots representing cc disperssion against multiple randomizations
estRank = estimateRank(data_matrix, nb_permutation = 40, interval = seq(2,9))
rank = 2
#Apply sNMF to our new data
fit_sNMF= nmf(t(data_matrix),rank, method='snmf/r', nrun=nrun, beta = beta)

#predict clusters from consensus matrix
clusters=predict(fit_sNMF, what= 'consensus')


#evaluation mesures 
#Silhouette 
si=silhouette(fit_sNMF, what= 'consensus')
plot(si)
#CC 
cophcor(fit_sNMF)

#extract features -top two features) 
features = extractFeatures(basis(fit_sNMF), 2L)
top_path = vector(mode = "character", length = 20)
jj = 1
for(i in 1:5)
{
  top_path[jj] = as.character(i)
  jj= jj+1
  for (j in 1:length(features[[i]]))
  {
    top_path[jj] = as.character(pathway_names[features[[i]][j],1])
    jj= jj+1
  }
}

# clinical and biological relevence analysis

#categorical variables x clusters  (chi^2 test)
chisq.test(table(clusters,clinical_data$gender))

#Numerical variables x clusters (anova test)
anova(lm(clinical_data$Age) ~ clusters))

#survival analysis
survdiff(Surv(as.numeric(survival_data[,2]),as.numeric(survival_data[,3])) ~ clusters)
#Kaplan meier curves
fit_s=survfit(Surv(survival_data$OS,survival_data$VITAL_STATUS) ~ clusters)
plot(fit_s,col=c(1:4),xlab = "time",ylab = "survival probability")
legend ("topright",c("subtype1","subtype2"), col=(1:4), lwd=0.5,cex=0.8)

#correction of Pèvalue of survival analysis (if correlation between age and clusters is proved)
# aply likelihood ratio test 
null_model= coxph(Surv(survival_data$OS,survival_data$VITAL_STATUS) ~ clinical_data$Age)
alternative_model= coxph(Surv(survival_data$OS,survival_data$VITAL_STATUS) ~ clinical_data$Age + clusters)
anova(null_model, alternative_model)

#Mutation enrichment (fisher exact test)
len= dim(mutation_data)[2] 

mutation<- vector(mode="character", length=len)
cluster1=  vector(mode="double", length=len)
cluster2 =  vector(mode="double", length=len)
test = data.frame(mutation,cluster1,cluster2)
test$mutation= ""

#create adjacency matrix
for(j in 1:len)
{
  test$mutation[j]= colnames(mutation_data)[j]
  successsample = length(which(mutation_data[which(clusters == 1),j]==1))
  successleftpart=  length(which(mutation_data[,j]==1))-successsample
  failuresamples=length(which(clusters == 1)) - successsample
  failureleftpart = length(which(mutation_data[,j]==0))-failuresamples
  test$cluster1[j] = fisher.test(matrix(c(successsample, successleftpart,failuresamples, failureleftpart), 2, 2), alternative='greater')$p.value; 
  
  
}
#BH correction
test$cluster1 = p.adjust(test$cluster1 , 'BH')


for(j in 1:len)
{
  successsample = length(which(mutation_data[which(clusters == 2),j]==1))
  successleftpart=  length(which(mutation_data[,j]==1))-successsample
  failuresamples=length(which(clusters == 2)) - successsample
  failureleftpart = length(which(mutation_data[,j]==0))-failuresamples
  test$cluster2[j] = fisher.test(matrix(c(successsample, successleftpart,failuresamples, failureleftpart), 2, 2), alternative='greater')$p.value; 
  
  
}
test$cluster2 = p.adjust(test$cluster2 , 'BH')
enrichment = subset(test, mutation %in% test$mutation[which(test$cluster1 <0.05)] 
|        mutation %in% test$mutation[which(test$cluster2 <0.05)] )
              
