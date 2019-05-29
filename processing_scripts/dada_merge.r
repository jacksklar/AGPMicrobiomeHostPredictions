library(dada2);library(digest)

outputpath<-"/data/sklarjg/AmericanGutProj/Merged_Data/"

tab1<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/HiSeq2500_TINY/HiSeq2500_TINY_seqtab.rds")
tab2<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/HiSeq2500_TRIM/HiSeq2500_TRIM_seqtab.rds")
tab3<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/MiSeq1/MiSeq1_seqtab.rds")
tab4<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/MiSeq2/MiSeq2_seqtab.rds")
tab5<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/MiSeq3/MiSeq3_seqtab.rds")
tab6<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/MiSeq4/MiSeq4_seqtab.rds")
tab7<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/MiSeq5/MiSeq5_seqtab.rds")
tab8<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/MiSeq6/MiSeq6_seqtab.rds")
tab9<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/MiSeq7/MiSeq7_seqtab.rds")
tab10<-readRDS("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/MiSeq8/MiSeq8_seqtab.rds")

merge <- mergeSequenceTables(tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10)

saveRDS(merge, paste(outputpath,"merged.rds",sep=""))

blooms<-rownames(read.table(file="/data/sklarjg/AmericanGutProj/bloom_taxa_fna.txt",sep="\t",row.names=1))
tab4<-merge[,!colnames(merge)%in%blooms]


taxa <- assignTaxonomy(tab4, "/data/sklarjg/AmericanGutProj/silva_nr_v128_train_set.fa.gz", multithread=TRUE)
saveRDS(taxa, paste(outputpath,"merged_taxa_noblm.rds",sep=""))


colnames(tab4)<-sapply(colnames(tab4), digest, algo="md5")
write.table(t(tab4),paste(outputpath,"merged_seqtab_md5.xls",sep=""),sep="\t",col.names=NA)


taxamd5<-taxa
rownames(taxamd5)<-sapply(rownames(taxamd5), digest, algo="md5")
write.table(cbind(taxamd5, rownames(taxa)), paste(outputpath,"taxa_md5.xls",sep=""),sep="\t",col.names=NA)




