library(graphite)
#load NCI pathways
getpaths <- pathways("hsapiens", "nci")
mapGeneEntrez= convertIdentifiers(getpaths, "entrez")
mapGeneSymbol= convertIdentifiers(getpaths, "symbol")

# load our datasets mRNA,miRNA and DNA methylation 
mapping1 <- function(col_name) {
  sym_ID = matrix(ncol = 1, nrow=length(col_name))
  for (i in 1 : length(col_name))
  { ch = unlist(strsplit(col_name[i], "[|]"))
  sym_ID[i]= ch[1]
  }
  return (sym_ID)
}

#start example 

load(GBM_ProcessedData)
miRNA_target= read.csv( 'E:/newflder/data/mirTarget.csv',header=FALSE)
gene_comp= t(read.csv( 'E:/newflder/data/gene_comp.csv',header=FALSE))

#reshape gene_comp as a table of two columns
g_comp = matrix(ncol = 2, nrow=300000)
len = 0
j= 1
for (i in 2:nrow(gene_comp) )
{ 
  g_c= gene_comp[i,2:length(gene_comp[i,])]
  g_c = g_c[!g_c == "" ]
  len = len +length(g_c)
  g_comp[j:len,1] = gene_comp[i,1]
  g_comp[j:len,2] = g_c
  j= j+length(g_c)
}
g_comp= g_comp[complete.cases(g_comp), ]

#miRNA from miRNA_target set
target=as.character(miRNA_target[,1])
#miRNA features
miRNAcomponents = colnames(miRNASEQ)
miRNAcomponents=gsub("r","R",as.character(miRNAcomponents))
# mRNA features
genes= colnames(RNASEQ)
genes= mapping1(genes)
Methy_components= colnames(mydatME)
genecnv = colnames(CNV)
#tables of 3 columns; Columns1:pathways,Columns2:omics type, Columns3:mappedcomponent 
selected_features = matrix(ncol = 3, nrow=300000)
len = 0
j= 1
indx = NULL
path_sym=c("SLC39A4","STAT3", "CDK4","CDK6","CDK2B","CDK2A")
path_Entrez = c(0)
chosenpath= c(188,198,60,71,103,126,127,6,56,168,206,95,146)
#chosenpath= c(126,6,198,206)

len = 0
j= 1
indx = NULL
for (i in 1:  length(mapGeneSymbol@entries)) {
  #get pathway genes as Symbols 
  gsym <- pathwayGraph(mapGeneSymbol@entries[[i]])
  path_sym = nodes(gsym)
  path_sym = gsub("SYMBOL:", "", path_sym)
  #get pathway genes as entrez IDs   
  gEntrez <- pathwayGraph(mapGeneEntrez@entries[[i]])
  path_Entrez = nodes(gEntrez)
  path_Entrez = gsub("ENTREZID:", "", path_Entrez)
  
  #start mapping ...
  #mapping components of the first dataset: mRna
  j = j+length(indx)
  indx = NULL
  indx = which( genes%in% path_sym)
  if (length(indx) != 0){
    
    len = len +length(indx)
    selected_features[j:len,1]= i
    selected_features[j:len,2]= 1
    selected_features[j:len,3]= indx}
  # mapping CNV
  
  j = j+length(indx)
  indx = NULL
  indx = which( genecnv%in% path_sym)
  if (length(indx) != 0){
    
    len = len +length(indx)
    selected_features[j:len,1]= i
    selected_features[j:len,2]= 2
    selected_features[j:len,3]= indx}
  
  # #mapping components of the second dataset: DNA methylation
  # j = j+length(indx)
  # indx = NULL
  # indx = which(Methy_components %in% 
  #                unique(as.character(gene_comp[which(gene_comp[,1] 
  #                                                    %in% path_sym),2])))
  # 
  # 
  # if (length(indx) != 0){
  #   len = len +length(indx)
  #   selected_features[j:len,1]= i
  #   selected_features[j:len,2]= 2
  #   selected_features[j:len,3]= indx}  
  # 
  #mapping components of the third dataset: miRna
  j = j+length(indx)
  indx = NULL
  indx = which(miRNAcomponents %in% 
                 unique(as.character(target[which(miRNA_target[,4] 
                                                  %in% path_Entrez)])))
  
  if (length(indx) != 0){
    len = len +length(indx)
    selected_features[j:len,1]= i
    selected_features[j:len,2]= 3
    selected_features[j:len,3]= indx}
  
}
mapping_result= selected_features[complete.cases(selected_features), ]
write.csv(mapping_result, 'F:/brcadata/selected_features.csv',row.names=FALSE)
