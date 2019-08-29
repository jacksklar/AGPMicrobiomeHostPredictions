library(dada2);library(digest)

outputpath<-"/data/sklarjg/AmericanGutProj/Merged_Data/"

merge<-readRDS("/data/sklarjg/AmericanGutProj/Merged_Data/merged.rds")

tab3<-merge[,1:100000]
write.table(t(tab3),paste(outputpath,"merged_seqtab_top100k.xls",sep=""),sep="\t",col.names=NA)

blooms<-rownames(read.table(file="/data/sklarjg/AmericanGutProj/bloom_taxa_fna.txt",sep="\t",row.names=1))
tab4<-tab3[,!colnames(tab3)%in%blooms]
write.table(t(tab4),paste(outputpath,"merged_seqtab_noblm_top100k.xls",sep=""),sep="\t",col.names=NA)

taxa <- assignTaxonomy(tab4, "/data/sklarjg/AmericanGutProj/silva_nr_v128_train_set.fa.gz", multithread=TRUE)
saveRDS(taxa, paste(outputpath,"merged_taxa_noblm_top100k.rds",sep=""))

taxamd5<-taxa
rownames(taxamd5)<-sapply(rownames(taxamd5), digest, algo="md5")
write.table(cbind(taxamd5, rownames(taxa)), paste(outputpath,"taxa_md5.xls",sep=""),sep="\t",col.names=NA)

md5s<-sapply(colnames(tab4), digest, algo="md5")
write.table(md5s,paste(outputpath,"merged_md5_top100k.xls",sep=""),sep="\t")
colnames(tab4)<-md5s
write.table(t(tab4),paste(outputpath,"merged_seqtab_md5_top100k.xls",sep=""),sep="\t",col.names=NA)




