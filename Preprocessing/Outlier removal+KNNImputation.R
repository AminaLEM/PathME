#Data are downloaded from https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018
CNV = read.table( "E://pancancer/data_CNA.txt", sep= "\t")
RNASeq = read.table( "E://pancancer/data_RNA_Seq_v2_expression_median.txt", sep= "\t")

CNV =t(CNV)
RNASeq = t(RNASeq)

colnames(CNV)= CNV[2,]
rownames(CNV)= CNV[,1]
CNV = CNV[c(-1,-2),-1]

colnames(RNASeq)= RNASeq[2,]
rownames(RNASeq)= RNASeq[,1]
RNASeq = RNASeq[c(-1,-2),-1]

rnaseqq = RNASeq[,colSums(is.na(RNASeq))<nrow(RNASeq)*0.2]
rnaseqqq = rnaseqq[rowSums(is.na(rnaseqq))<ncol(rnaseqq)*0.2,]

cnvv = CNV[,colSums(is.na(CNV))<nrow(CNV)*0.2]
cnvvv = cnvv[rowSums(is.na(cnvv))<ncol(cnvv)*0.2,]


sharedcases = rownames(cnvvv)[which(rownames(cnvvv) %in% rownames(rnaseqqq))]


rnaseqpre = impute.knn(t(rnaseqqq))
rnaseq= t(rnaseqpre$data)


cnvpre = impute.knn(t(cnvvv))
cnv= t(cnvpre$data)
