#Violin plot mutational burden by pathway (the top selected pathways) and by the resulted subtypes, 
library(ggplot2)
library(gridExtra)

ToothGrowth1= read.csv('F:/brcadata/89brca.csv', sep =',', header = T)

p1 <- ggplot(ToothGrowth1, aes(x=ToothGrowth1[,2], y=ToothGrowth1[,1])) + 
  geom_violin()
SoilSciGuylabs <- c("Subtype1","Subtype2","Subtype3","Subtype4","Subtype5")
pp1=p1+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title=" Canonical Wnt signaling pathway ",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)


ToothGrowth2= read.csv('F:/brcadata/58brca.csv', sep =',', header = T)

p2 <- ggplot(ToothGrowth2, aes(x=ToothGrowth2[,2], y=ToothGrowth2[,1])) + 
  geom_violin()
pp2=p2+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title="IL23-mediated signaling events",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)


ToothGrowth3= read.csv('F:/brcadata/10brca.csv', sep =',', header = T)

p3 <- ggplot(ToothGrowth3, aes(x=ToothGrowth3[,2], y=ToothGrowth3[,1])) + 
  geom_violin()
pp3=p3+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title="PDGFR-beta signaling pathway",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)

ToothGrowth4= read.csv('F:/brcadata/26brca.csv', sep =',', header = T)

p4 <- ggplot(ToothGrowth4, aes(x=ToothGrowth4[,2], y=ToothGrowth4[,1])) + 
  geom_violin()
pp4=p4+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title=" Regulation of Ras family activation",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)



ToothGrowth5= read.csv('F:/brcadata/18brca.csv', sep =',', header = T)

p5 <- ggplot(ToothGrowth5, aes(x=ToothGrowth5[,2], y=ToothGrowth5[,1])) + 
  geom_violin()
pp5=p5+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title=" Canonical Wnt signaling pathway ",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)


ToothGrowth6= read.csv('F:/brcadata/52brca.csv', sep =',', header = T)

p6 <- ggplot(ToothGrowth6, aes(x=ToothGrowth6[,2], y=ToothGrowth6[,1])) + 
  geom_violin()
pp6=p6+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title="IL23-mediated signaling events",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)


ToothGrowth7= read.csv('F:/brcadata/84brca.csv', sep =',', header = T)

p7 <- ggplot(ToothGrowth7, aes(x=ToothGrowth7[,2], y=ToothGrowth7[,1])) + 
  geom_violin()
pp7=p7+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title="PDGFR-beta signaling pathway",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)

ToothGrowth8= read.csv('F:/brcadata/33brca.csv', sep =',', header = T)

p8 <- ggplot(ToothGrowth8, aes(x=ToothGrowth8[,2], y=ToothGrowth8[,1])) + 
  geom_violin()
pp8=p8+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title=" Regulation of Ras family activation",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)



ToothGrowth9= read.csv('F:/brcadata/115brca.csv', sep =',', header = T)

p9 <- ggplot(ToothGrowth9, aes(x=ToothGrowth9[,2], y=ToothGrowth9[,1])) + 
  geom_violin()
pp9=p9+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title="PDGFR-beta signaling pathway",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)

ToothGrowth10= read.csv('F:/brcadata/94brca.csv', sep =',', header = T)

p10 <- ggplot(ToothGrowth10, aes(x=ToothGrowth10[,2], y=ToothGrowth10[,1])) + 
  geom_violin()
pp10=p10+ geom_jitter(shape=16, position=position_jitter(0.2)) + labs(title=" Regulation of Ras family activation",x="Subtypes", y = "Mutational burden")+scale_x_discrete(labels= SoilSciGuylabs)


grid.arrange(pp1, pp2,pp3,pp4, nrow = 2)
grid.arrange(pp5, pp6,pp7,pp8, nrow = 2)
grid.arrange(pp9, pp10, nrow = 2)
