library(lme4,lib.loc='../r_packages')

setwd('/udd/reema/Postdoc/Progetti/COPDGENE/copd_vae')

#main_dir = '/udd/reema/Postdoc/Progetti/COPDGENE/copd_vae/copd_vae/gendata/prospective'

main_dir = 'copd_vae/gendata_1/prospective'

data <- read.table(paste0(main_dir, "/lfu_data.tsv"),colClasses = c('sid'='factor','branch'='factor'))
data$branch <- relevel(data$branch,ref=5)

# Fixed effect Poisson regression

glm_model <- glm(Net_Exacerbations ~ branch + age + sex + race, data=data, family = poisson(link = "log"))
glm_coef <- summary(glm_model)$coefficients
write.table(glm_coef,paste0(main_dir, '/exacerbations_glm_coef.tsv'),sep='\t')
sink(paste0(main_dir, '/exacerbations_glm_summary.tsv'))
print(summary(glm_model))
sink()

# Mixed effect Poisson regression

glmer_model <- glmer(Net_Exacerbations ~ branch + age + sex + race + (1|sid), data = data, family = poisson(link = "log"))
glmer_coef <- summary(glmer_model)$coefficients
write.table(glmer_coef,paste0(main_dir, '/exacerbations_glmer_coef.tsv'),sep='\t')
sink(paste0(main_dir, "/exacerbations_glmer_summary.tsv"))
print(summary(glmer_model))
sink()
