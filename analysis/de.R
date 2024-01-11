library(DESeq2)
library("magrittr")
library(dplyr)
library(stringr)
library(edgeR)
library(EnsDb.Hsapiens.v79)

expr_dir <- "gendata_2/data/"
de_dir <- "gendata_2/de/"
branches_dir <- "gendata_2/elpi/"

counts_raw = read.table(file.path(expr_dir,"counts_raw.tsv"),sep='\t',header=TRUE, row.names=1)
counts_raw <- as.matrix(counts_raw)
counts <- round(counts_raw)
colnames(counts) <- substring(colnames(counts),2)
#
#
masterfile <- read.table(file.path(expr_dir,'masterfile.tsv'),sep='\t',header=TRUE, row.names=1)
pheno <- read.table(file.path(de_dir,'pheno_branches_covariates_cellcov.tsv'),sep='\t',header=TRUE, row.names=1) ##Already ordered based on the rows of branches.tsv
branches <- read.table(file.path(branches_dir,'branches.tsv'),sep='\t',header=TRUE, row.names=1)
#
#
counts <- counts[,rownames(branches)]
masterfile <- masterfile[rownames(branches),]
#
#
d0 <- DGEList(counts)
d0 <- calcNormFactors(d0, method = "TMM")
keep <- rowSums(cpm(d0$counts) >= 1) >= 10
d <- d0[keep, , keep.lib.sizes=FALSE]
#
#
pheno$batch <- as.factor(masterfile$Lc.Batch)
pheno$batch <- as.factor(str_replace(pheno$batch,"-","."))
pheno[,"branch"] <- as.factor(branches$branch)
#
quantb1 <- quantile(branches[branches$branch==1,"branch_pos"],0.5)
quantb3 <- quantile(branches[branches$branch==3,"branch_pos"],0.5)
quantb4 <- quantile(branches[branches$branch==4,"branch_pos"],0.5)
#
branchc1 <- branches$branch
branchc1[(branches$branch==1) & (branches$branch_pos>=quantb1)] <- "1high"
branchc1[(branches$branch==1) & (branches$branch_pos<quantb1)] <- "1low"
branchc1[(branches$branch==3) & (branches$branch_pos>=quantb3)] <- "3high"
branchc1[(branches$branch==3) & (branches$branch_pos<quantb3)] <- "3low"
branchc1[(branches$branch==4) & (branches$branch_pos>=quantb4)] <- "4high"
branchc1[(branches$branch==4) & (branches$branch_pos<quantb4)] <- "4low"
pheno[,"branch.percentile"] <- as.factor(branchc1)
#
branchc2 <- branches$branch
branchc2[(branchc2==5) | (branchc2==6)] <- "healthy"
branchc2[(branches$branch==1) & (branches$branch_pos>=quantb1)] <- "1high"
branchc2[(branches$branch==1) & (branches$branch_pos<quantb1)] <- "1low"
branchc2[(branches$branch==3) & (branches$branch_pos>=quantb3)] <- "3high"
branchc2[(branches$branch==3) & (branches$branch_pos<quantb3)] <- "3low"
branchc2[(branches$branch==4) & (branches$branch_pos>=quantb4)] <- "4high"
branchc2[(branches$branch==4) & (branches$branch_pos<quantb4)] <- "4low"
pheno[,"branch.healthy"] <- as.factor(branchc2)
#
#
age_col <- pheno$Age_P2
batch_col <- pheno$batch
race_col <- pheno$race
gender_col <- pheno$gender
neutrophl_col <- pheno$neutrophl_pct_P2
lymphcyt_col <- pheno$lymphcyt_pct_P2
monocyt_col <- pheno$monocyt_pct_P2
eosinphl_col <- pheno$eosinphl_pct_P2
basophl_col <- pheno$basophl_pct_P2
branch_col <- pheno$branch
branch_percentile_col <- pheno$branch.percentile
branch_healthy_col <- pheno$branch.healthy
#
#
#Fits and comparisons
mm <- model.matrix(~ 0 + branch_percentile_col + batch_col + age_col + race_col + gender_col + neutrophl_col + lymphcyt_col + monocyt_col + eosinphl_col)
y <- voom(d, mm, plot = T)
fit <- lmFit(y, mm)
fit <- eBayes(fit)
contr_cols = c("branch_percentile_col1high - (branch_percentile_col5+branch_percentile_col6)/2","branch_percentile_col3high - (branch_percentile_col5+branch_percentile_col6)/2","branch_percentile_col4high - (branch_percentile_col5+branch_percentile_col6)/2","branch_percentile_col1high - branch_percentile_col5","branch_percentile_col1high - branch_percentile_col6","branch_percentile_col3high - branch_percentile_col5","branch_percentile_col3high - branch_percentile_col6","branch_percentile_col4high - branch_percentile_col5","branch_percentile_col4high - branch_percentile_col6")
contr <- makeContrasts(contrasts = contr_cols, levels = colnames(coef(fit)))
cfit <- contrasts.fit(fit, contr)
cfit <- eBayes(cfit)

mm1 <- model.matrix(~ 0 + branch_healthy_col + batch_col + age_col + race_col + gender_col + neutrophl_col + lymphcyt_col + monocyt_col + eosinphl_col)
y1 <- voom(d, mm1, plot = T)
fit1 <- lmFit(y1, mm1)
fit1 <- eBayes(fit1)
contr_cols1 = c("branch_healthy_col1high - branch_healthy_colhealthy","branch_healthy_col3high - branch_healthy_colhealthy","branch_healthy_col4high - branch_healthy_colhealthy", "branch_healthy_col1high - branch_healthy_col3high", "branch_healthy_col1high - branch_healthy_col4high","branch_healthy_col3high - branch_healthy_col4high")
contr1 <- makeContrasts(contrasts = contr_cols1, levels = colnames(coef(fit1)))
cfit1 <- contrasts.fit(fit1, contr1)
cfit1 <- eBayes(cfit1)


joined_df <- topTable(cfit,number=Inf,coef=1)
colnames(joined_df) <- paste(colnames(joined_df), str_remove_all(contr_cols[1], "branch_percentile_"), sep="_")
joined_df <- cbind(genesID = rownames(joined_df), joined_df)

for (x in 2:length(contr_cols)){
	contrastname <- str_remove_all(contr_cols[x], "branch_percentile_")
	singletab <- topTable(cfit,number=Inf,coef=x)
	colnames(singletab) <- paste(colnames(singletab), contrastname, sep="_")
	singletab <- cbind(genesID = rownames(singletab), singletab)
	joined_df <- merge(joined_df, singletab,by.x = "genesID", by.y = "genesID")
}

for (x in 1:length(contr_cols1)){
	contrastname <- str_remove_all(contr_cols1[x], "branch_healthy_")
	singletab <- topTable(cfit1,number=Inf,coef=x)
	colnames(singletab) <- paste(colnames(singletab), contrastname, sep="_")
	singletab <- cbind(genesID = rownames(singletab), singletab)
	joined_df <- merge(joined_df, singletab,by.x = "genesID", by.y = "genesID")
}

colnames(joined_df) <- str_remove_all(colnames(joined_df)," ")
write.table(joined_df,	file.path(de_dir,'joined_DE_cellcov_nobasophl.tsv'),sep = '\t',quote=FALSE)











