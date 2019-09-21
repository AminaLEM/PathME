# gotten from https://github.com/CamilaDuitama/MasterThesis/blob/master/DownloadTCGA.R
library(RTCGAToolbox)
library(RTCGA)
library(TCGAbiolinks)
library(SummarizedExperiment)

####Using RTCGAToolbox
stddata <- getFirehoseRunningDates()
stddata
# clinical,miRNASeqGene(0*0), RNASeq2GeneNorm(0*0),Methylation,
#mRNAArray,miRNAArray,GISTIC,RNAseqNorm,RNAseq2Norm
GBMData <- getFirehoseData (dataset="GBM", runDate="20160128",
                            clinic=TRUE,mRNASeq= TRUE, mRNAArray=TRUE,miRNAArray=TRUE,Methylation=TRUE,
                            GISTIC =TRUE, RNAseqNorm= TRUE, RNAseq2Norm=TRUE,miRNASeqGene=TRUE,CNASNP=TRUE,
                            CNASeq= TRUE, CNACGH = TRUE, Mutation = TRUE)
Clinical <- getData(GBMData,"clinical")
#mRNA microArray platform: Affymetrix Human Genome U133 Array
mRNAArray <- getData(GBMData,"mRNAArray",platform = 3)
CopyNumber<-getData(GBMData,"GISTIC","AllByGene")
Methylation<-getData(GBMData,"Methylation",platform=1)
miRNAArray<-getData(GBMData,"miRNAArray",1)
SomaticMutation <- getData(GBMData,"Mutation")

write.table(Clinical, file = "Data/RTCGAToolbox/Clinical.csv")
write.table(mRNAArray,file="Data/RTCGAToolbox/mRNAArray.csv")
write.table(CopyNumber,file="Data/RTCGAToolbox/CopyNumber.csv")
write.table(Methylation,file="Data/RTCGAToolbox/Methylation.csv")
write.table(miRNAArray,file="Data/RTCGAToolbox/MiRNAArray.csv")
write.table(SomaticMutation,file="Data/RTCGAToolbox/SomaticMutation.csv")

#With TCGABiolinks 
#Note: TCGABiolinks has access to 2 sources
#GDC Legacy Archive : provides access to an unmodified copy of data that was previously
#stored in CGHub and in the TCGA Data Portal hosted by the TCGA Data Coordinating Center 
#(DCC), in which uses as references GRCh37 (hg19) and GRCh36 (hg18).
#GDC harmonized database: data available was harmonized against GRCh38 (hg38) using GDC
#Bioinformatics Pipelines which provides methods to the standardization of biospecimen 
#and clinical data.
### ATTENTION: We only use Legacy data, because there was more mRNA data 
Clinical.1<-GDCquery_clinic(project="TCGA-GBM",type="Clinical")
Methylation.query<-GDCquery(project= "TCGA-GBM", 
                             data.category = "DNA methylation", 
                             platform = "Illumina Human Methylation 450", legacy =TRUE)
GDCdownload(Methylation.query)
Methylation.1 <- GDCprepare(Methylation.query)

write.table(Clinical.1, file = "Data/TCGABiolinks/Clinical.csv")
write.table(mRNAArray.1,file="Data/TCGABiolinks/mRNAArray.csv")
write.table(CopyNumber.1,file="Data/TCGABiolinks/CopyNumber.csv")
write.table(Methylation.1,file="Data/TCGABiolinks/Methylation.csv")
write.table(miRNAArray.1,file="Data/TCGABiolinks/MiRNAArray.csv")
write.table(SomaticMutation.1,file="Data/TCGABiolinks/SomaticMutation.csv")


#Code to use multiMiR to get the validated targets of a list of miRNA
library(multiMiR)
library(readr)
miRNA_GBM_Firebrowse <- read_csv("miRNA_GBM_Firebrowse.csv", 
                                   +     col_names = FALSE)
result<-list()
for (i in names(miRNA_GBM_Firebrowse))
{
  example1 <- get_multimir(mirna = i, summary = TRUE)
  result <- c(result, example1@data$target_symbol) 
}
result<-unlist(result, recursive=FALSE)
write.table(result,file="miRNA_targets_GBM_Firebrowse.csv", row.names = FALSE, col.names = FALSE, sep = ",")

