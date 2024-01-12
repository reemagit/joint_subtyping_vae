args <- commandArgs(trailingOnly = TRUE)
gendata_dir <- file.path(args[1]) 
data_dir <- file.path(args[1], "data")
out_dir <- file.path(args[1], "prospective")


#main_dir = '/udd/reema/Postdoc/Progetti/COPDGENE/copd_vae/copd_vae/gendata/prospective'

main_dir = gendata_dir

data <- read.table(file.path(data_dir, "lfu_covariates.tsv"),colClasses = c('sid'='factor','branch'='factor'))
data$branch <- relevel(data$branch,ref=5)

# Fixed effect Poisson regression

glm_model <- glm(Net_Exacerbations ~ branch + age + sex + race, data=data, family = poisson(link = "log"))
glm_coef <- summary(glm_model)$coefficients
write.table(glm_coef,file.path(out_dir, 'exacerbations_glm_coef.tsv'),sep='\t')
sink(file.path(out_dir, 'exacerbations_glm_summary.tsv'))
print(summary(glm_model))
sink()

# Mixed effect Poisson regression

glmer_model <- glmer(Net_Exacerbations ~ branch + age + sex + race + (1|sid), data = data, family = poisson(link = "log"))
glmer_coef <- summary(glmer_model)$coefficients
write.table(glmer_coef,file.path(out_dir, 'exacerbations_glmer_coef.tsv'),sep='\t')
sink(file.path(out_dir, "exacerbations_glmer_summary.tsv"))
print(summary(glmer_model))
sink()
