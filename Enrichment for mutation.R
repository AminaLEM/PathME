#Build contingency matrix
build_contingency=  function(Xmut, group){
   
   successsample = length(which(Xmut[group]==1))
   successleftpart=  length(which(Xmut==1))-successsample
   failuresamples=length(group)- successsample
   failureleftpart = length(which(Xmut==0))-failuresamples
   
   contingency_table= matrix(c(successsample, successleftpart,failuresamples, failureleftpart), 2, 2)
  return (contingency_table)
 }
 # Extract rows that contain values < 0.05 to select mutations overpresented in the clustering
 findindexes =  function(arr){
   leng=dim(arr)[2]
   res =  list()
 for (i in 1:leng)
   res= append (res, which(arr[,i] < 0.05))
 return (res)
 }
 # construct the array of mutations* clusters that contains P-values from fisher exact test with BH correction
 #it returns mutation overrepresented in certain clusters
 mutation_enrichment =  function(mutations, groups)
 {
  en= dim(mutations)[2] 
  enrichment=colnames(mutations)
  
    #fisher test and p-value correction for all groups
  grp=length(unique(groups))
  for (n in 1:grp)
  {
  cluster=list()
  len = dim(mutations)[2]
  for (j in 1:len)
  {
  contingency_table= build_contingency(mutations[,j], which(groups==n))
  cluster[j]= fisher.test(contingency_table, alternative='greater')$p.value
  }
  enrichment= cbind(enrichment,p.adjust(cluster , 'BH'))
  }
  grps= grp+1
  enri= enrichment[,2:grps]
  class(enri)= "numeric"
  extractedMut= enrichment[unlist(findindexes(enri)),]
  colnames(extractedMut)= c("mutations",1:grp)
  return (extractedMut)
  }
