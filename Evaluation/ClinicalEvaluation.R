# BRCA LIU endpoints

endpoints = read.csv("G:/brcadata/clinicaledpointBRCA.csv", sep =",", header = T)
endpoints = endpoints[which(endpoints$bcr_patient_barcode%in% clinical_data$bcr_patient_barcode),]

# GMB LIU endpoints
samples = rownames(clinicda)
endpoints = read.csv("G:/gbm/endpoints.csv", sep =";", header = T)
endpoints = endpoints[which(endpoints$bcr_patient_barcode %in% rownames(clinicda)),]
endpoints2=endpoints

for (i in 1:273)
{
  endpoints2[i,] = endpoints[which(endpoints$bcr_patient_barcode == samples[i]),]
}
endpoints= endpoints2

#LUSC LIU endpoints

samples = substr(survival_data[,1],1,12)
endpoints = read.csv("G:/lusc/endpoints.csv", sep =";", header = T)
endpoints = endpoints[which(endpoints$bcr_patient_barcode %in% samples),]
endpoints2=endpoints

for (i in 1:106)
{
  endpoints2[i,] = endpoints[which(endpoints$bcr_patient_barcode == samples[i]),]
}
endpoints= endpoints2


#CRC LIU endpoints

samples = toupper(gsub("\\.","-",clinc[,1]))
endpoints = read.csv("G:/crc/endpoints.csv", sep =";", header = T)
endpoints = endpoints[which(endpoints$bcr_patient_barcode %in% samples),]
endpoints2=rbind(endpoints, endpoints[1,])

for (i in 1:294)
{
  print(i)
  if(exists(samples[i], where=as.matrix(endpoints$bcr_patient_barcode)))
  endpoints2[i,] = endpoints[which(endpoints$bcr_patient_barcode == samples[i]),]
  else
    endpoints2[i,] = rep(NA,26)
}
endpoints= endpoints2


# clinical and biological relevence analysis based on LIU et al. clinical data
clinical_data = clinical_data[which(as.matrix(clinical_data[,2])%in% as.matrix(samples)[,1]),]
#categorical variables x clusters  (chi^2 test)
chisq.test(table(clusters,factor(clinical_data$gender)))
indx=which(as.matrix(clinical_data$race) %in% c("[Not Evaluated]","[Not Available]"))
chisq.test(table(clusters[-indx],as.matrix(clinical_data$race)[-indx]))
indx=which(as.matrix(clinical_data$ajcc_pathologic_tumor_stage) %in% c("[Discrepancy]","[Not Available]"))
chisq.test(table(clusters[-indx],as.matrix(clinical_data$ajcc_pathologic_tumor_stage)[-indx]))
chisq.test(table(clusters,as.matrix(clinical_data$histological_type)))
indx=which(as.matrix(clinical_data$tumor_status) %in% c("#N/A"))
chisq.test(table(clusters[-indx],as.matrix(clinical_data$tumor_status)[-indx]))
indx=which(as.matrix(clinical_data$menopause_status) %in% c("[Not Evaluated]","[Unknown]", "[Not Available]"))
chisq.test(table(clusters,clinical_data$menopause_status))
indx=which(as.matrix(clinical_data$new_tumor_event_site_other) %in% c("#N/A"))
chisq.test(table(clusters[-indx],as.matrix(clinical_data$new_tumor_event_site)[-indx]))

#Numerical variables x clusters (anova test)
indx=which(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis) %in% c("#N/A"))
anova(lm(as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)) ~ clusters))
#survival analysis
indx=which(as.matrix(clinical_data$DSS) %in% c("#N/A"))
survdiff(Surv(as.numeric(clinical_data$OS.time),as.numeric(clinical_data$OS)) ~ clusters)
survdiff(Surv(as.numeric(clinical_data$DSS.time),as.numeric(clinical_data$DSS)) ~ clusters)
survdiff(Surv(as.numeric(as.matrix(clinical_data$DFI.time)),as.numeric(as.matrix(clinical_data$DFI))) ~ clusters)
clusters=predict(res_fit[[3]], what= 'consensus')
survdiff(Surv(as.numeric(endpoints$OS.time),as.numeric(endpoints$OS)) ~ clusters)
null_model= coxph(Surv(as.numeric(as.matrix(endpoints$OS.time)),as.numeric(as.matrix(endpoints$OS))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$OS.time)),as.numeric(as.matrix(endpoints$OS))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)) + clusters)
anova(null_model, alternative_model)

#GBM 
null_model= coxph(Surv(as.numeric(as.matrix(endpoints$OS.time)),as.numeric(as.matrix(endpoints$OS))) ~ as.numeric(as.matrix(endpoints$age_at_initial_pathologic_diagnosis)))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$OS.time)),as.numeric(as.matrix(endpoints$OS))) ~ as.numeric(as.matrix(endpoints$age_at_initial_pathologic_diagnosis)) + clusters)
anova(null_model, alternative_model)

survdiff(Surv(as.numeric(endpoints$PFS.time),as.numeric(endpoints$PFS)) ~ clusters)
null_model= coxph(Surv(as.numeric(as.matrix(endpoints$PFS.time)),as.numeric(as.matrix(endpoints$PFS))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$PFS.time)),as.numeric(as.matrix(endpoints$PFS))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)) + clusters)
anova(null_model, alternative_model)

#GBM 
null_model= coxph(Surv(as.numeric(as.matrix(endpoints$PFS.time)),as.numeric(as.matrix(endpoints$PFS))) ~ as.numeric(as.matrix(clinicda[,15])))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$PFS.time)),as.numeric(as.matrix(endpoints$PFS))) ~ as.numeric(as.matrix(clinicda[,15])) + clusters)
anova(null_model, alternative_model)


survdiff(Surv(as.numeric(endpoints$PFI.time),as.numeric(endpoints$PFI)) ~ clusters)

null_model= coxph(Surv(as.numeric(as.matrix(endpoints$PFI.time)),as.numeric(as.matrix(endpoints$PFI))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$PFI.time)),as.numeric(as.matrix(endpoints$PFI))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)) + clusters)
anova(null_model, alternative_model)

#GBM 
null_model= coxph(Surv(as.numeric(as.matrix(endpoints$PFI.time)),as.numeric(as.matrix(endpoints$PFI))) ~ as.numeric(as.matrix(clinicda[,15])))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$PFI.time)),as.numeric(as.matrix(endpoints$PFI))) ~ as.numeric(as.matrix(clinicda[,15])) + clusters)
anova(null_model, alternative_model)

survdiff(Surv(as.numeric(endpoints$DSS.time),as.numeric(endpoints$DSS)) ~ clusters)

null_model= coxph(Surv(as.numeric(as.matrix(endpoints$DSS.time)),as.numeric(as.matrix(endpoints$DSS))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$DSS.time)),as.numeric(as.matrix(endpoints$DSS))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)) + clusters)
anova(null_model, alternative_model)

#GBM 
null_model= coxph(Surv(as.numeric(as.matrix(endpoints$DSS.time)),as.numeric(as.matrix(endpoints$DSS))) ~ as.numeric(as.matrix(clinicda[,15])))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$DSS.time)),as.numeric(as.matrix(endpoints$DSS))) ~ as.numeric(as.matrix(clinicda[,15])) + clusters)
anova(null_model, alternative_model)

survdiff(Surv(as.numeric(endpoints$DFI.time),as.numeric(endpoints$DFI)) ~ clusters)

null_model= coxph(Surv(as.numeric(as.matrix(endpoints$DFI.time)),as.numeric(as.matrix(endpoints$DFI))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)))
alternative_model= coxph(Surv(as.numeric(as.matrix(endpoints$DFI.time)),as.numeric(as.matrix(endpoints$DFI))) ~ as.numeric(as.matrix(clinical_data$age_at_initial_pathologic_diagnosis)) + clusters)
anova(null_model, alternative_model)
