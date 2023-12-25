library(DESeq2)
library(edgeR)
library(limma)

in_expr_dir <- 'data/expr/raw/'
in_pheno_dir <- 'data/pheno/processed/'

out_expr_dir <- 'data/expr/processed/'

counts <- read.table(paste0(in_expr_dir,'/counts_raw.tsv'),row.names = 1,header = TRUE)
counts <- as.matrix(counts)

counts2 <- round(counts)
colnames(counts2) <- substring(colnames(counts2),2)

masterfile <- read.table(paste0(in_expr_dir,'master.file.freeze4.txt'),sep='\t',header=TRUE)

masterfile <- masterfile[masterfile$final.analyzed == 1,]
masterfile <- masterfile[as.character(masterfile$actual_id) == as.character(masterfile$COPDgeneID),]
rownames(masterfile) <- masterfile$actual_id


pheno <- read.table(paste0(in_pheno_dir,'pheno.tsv'),sep='\t',header=TRUE,na.strings = "",fill=TRUE, quote = "",row.names = 'COPDgeneID')
pheno <- pheno[rownames(masterfile),]
pheno$gender <- as.factor(pheno$gender)
pheno$race <- as.factor(pheno$race)

masterfile$Lc.Batch <- as.factor(masterfile$Lc.Batch)

masterfile$gender <- pheno$gender
masterfile$race <- pheno$race
masterfile$age <- pheno$Age_P2


sids <- intersect(colnames(counts2),rownames(masterfile))



counts2 <- counts2[,sids]
masterfile <- masterfile[sids,]


dds <- DESeqDataSetFromMatrix(countData=counts2, colData=masterfile, design= ~ 1)

keep <- rowSums(cpm(counts(dds)) >= 1) >= 10

dds <- dds[keep,]

dds <- estimateSizeFactors(dds)
dds.log <- log2(counts(dds, normalized=TRUE)+1)

# LC adjusted data

dds.log.lc <- removeBatchEffect(dds.log, dds$Lc.Batch)
dds.log.lc.t <- t(dds.log.lc)

# LC sex age race adjusted data

dds.lc.sex.age.race <- dds[,!is.na(dds$gender) & !is.na(dds$race) & !is.na(dds$age) & !is.na(dds$Lc.Batch)]
dds.log.lc.sex.age.race <- log2(counts(dds.lc.sex.age.race, normalized=TRUE)+1)

mm <- model.matrix(~ gender + race + age + Lc.Batch, data=colData(dds.lc.sex.age.race),na.action="na.fail")

dds.log.lc.sex.age.race <- removeBatchEffect(dds.log.lc.sex.age.race, covariates = mm)

dds.log.lc.sex.age.race.t <- t(dds.log.lc.sex.age.race)

# Plot

# boxplot(dds.log.lc[,c(order(colSums(dds.log.lc),decreasing=T)[1:10],order(colSums(dds.log.lc),decreasing=F)[1:10])])

# Save

write.table(dds.log.lc.t, paste0(out_expr_dir,'deseq_counts_lc.tsv'),sep = '\t')
write.table(masterfile, paste0(out_expr_dir,'masterfile.tsv'),sep = '\t')
write.table(dds.log.lc.sex.age.race.t, paste0(out_expr_dir,'deseq_counts_lc_age_sex_race.tsv'),sep = '\t')
