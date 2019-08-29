# Import Dada2!
library(dada2); packageVersion("dada2")


# Subfolder containing batch of files

args <- commandArgs()
subfolder <- args[length(args)]
print(subfolder)
#subfolder <- "MiSeq1"

# Filename parsing
path <- paste("/data/sklarjg/AmericanGutProj/FASTQ/", subfolder, "/", sep="") 
filtpath <- paste("/data/sklarjg/AmericanGutProj/FILTERED_FASTQ/", subfolder, sep="")
outpath <- paste("/data/sklarjg/AmericanGutProj/DADA_OUTPUT_FILES/", subfolder, "/", sep="")


print(paste("FASTQ Path: ", path))
print(paste("Filtered FASTQ Path: ", filtpath))
print(paste("Output File Path: ", outpath))
fns <- list.files(path, pattern="fastq.gz") 


# Filtering
out<-filterAndTrim(file.path(path,fns), file.path(filtpath,fns), 
              truncLen=150, maxEE=1, truncQ=11, rm.phix=TRUE,
              compress=TRUE, verbose=TRUE, multithread=TRUE)
write.table(out,file=paste(outpath, subfolder,"_filter_out.csv",sep=""),sep=",",col.names=NA)           
   
   
# File parsing
filts <- list.files(filtpath, pattern="fastq.gz", full.names=TRUE) # CHANGE if different file extensions
sample.names <- sapply(strsplit(basename(filts), "_"), `[`, 1) # Assumes filename = sample_XXX.fastq.gz
names(filts) <- sample.names


# Learn error rates
set.seed(100)
err <- learnErrors(filts, nbases = 1e8, multithread=TRUE, randomize=TRUE)
pdf(file=paste(outpath, subfolder,"_plot_errors.pdf",sep=""))
plotErrors(err, nominalQ=TRUE)
dev.off()


# Infer sequence variants
dds <- vector("list", length(sample.names))
names(dds) <- sample.names
for(sam in sample.names) {
  cat("Processing:", sam, "\n")
  derep <- derepFastq(filts[[sam]])
  dds[[sam]] <- dada(derep, err=err, multithread=TRUE)
}


# Construct sequence table and write to disk
seqtab <- makeSequenceTable(dds)
saveRDS(seqtab, paste(outpath, subfolder,"_seqtab.rds",sep="")) # CHANGE ME to where you want sequence table saved
write.table(seqtab,file=paste(outpath, subfolder,"_seqtab.csv",sep=""),sep=",",col.names=NA)


# Assign taxonomy
tax <- assignTaxonomy(seqtab, "/data/sklarjg/AmericanGutProj/silva_nr_v128_train_set.fa.gz", multithread=TRUE)
saveRDS(tax, paste(outpath, subfolder,"_tax.rds",sep="")) # CHANGE ME ...
write.table(tax,file=paste(outpath, subfolder,"_tax.csv",sep=""),sep=",",col.names=NA)
st1 <- readRDS(paste(outpath, subfolder,"_seqtab.rds",sep=""))


# Remove chimeras
seqtab.nochim <- removeBimeraDenovo(st1, method="consensus", multithread=TRUE)
write.table(sum(seqtab.nochim)/sum(seqtab),file=paste(outpath, subfolder,"_seqtab.nochim_sumcomp.csv",sep=""),sep=",")


# Write to disk
saveRDS(seqtab.nochim, paste(outpath, subfolder,"_seqtab_nobimera.rds",sep="")) # CHANGE ME to where you want sequence table saved
write.table(seqtab.nochim,file=paste(outpath, subfolder,"_seqtab.nochim.csv",sep=""),sep=",",col.names=NA)

# FIN


