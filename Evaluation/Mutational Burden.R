library(graphite)
#load NCI pathways
getpaths <- pathways("hsapiens", "nci")
mapGeneEntrez= convertIdentifiers(getpaths, "entrez")
mapGeneSymbol= convertIdentifiers(getpaths, "symbol")

i= 97 #index of pathway
gsym <- pathwayGraph(mapGeneSymbol@entries[[iiiu]])
names(getpaths[i])
path_sym = nodes(gsym)
path_sym = gsub("SYMBOL:", "", path_sym)

# mutational burden by suptype
MutationalBurden1= rowSums(mutations[which (group == 1),colnames(mutations) %in% path_sym])/length(path_sym)
MutationalBurden2= rowSums(mutations[which (group == 2),colnames(mutations) %in% path_sym])/length(path_sym)
MutationalBurden3= rowSums(mutations[which (group == 3),colnames(mutations) %in% path_sym])/length(path_sym)
MutationalBurden4= rowSums(mutations[which (group == 4),colnames(mutations) %in% path_sym])/length(path_sym)
write.csv(rbind(as.matrix(MutationalBurden1),as.matrix(MutationalBurden2),as.matrix(MutationalBurden3),as.matrix(MutationalBurden4)), "D:/mb1.csv", row.names = F)
# MutationalBurden2[,2] contains the associated subtypes to samples
ktest= kruskal.test(  MutationalBurden2[,1] ~ MutationalBurden2[,2])
# p-values were then adjusted per desease using p.adjust({list of p-values associated to the top pathways} , 'BH')


