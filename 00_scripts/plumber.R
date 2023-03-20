
# Author: @Que
# Title: API Build - 3Sixty Health

library(plumber)
library(parsnip)
library(ranger)
library(h2o)
library(tidyverse)
library(rapidoc)

R.Version()
system('java --version')

# Not in container
library(recipes)
library(embed)

# Model to Run - Email Classification Features

# H2O Classification Analysis
#h2o.init()

### Load Production Model
#path <- file.path(rprojroot::find_rstudio_root_file(),
#                  "00_production_model/PROD_H2O_MODEL")

#h2o_model <- h2o.loadModel(path)

# Model to Run
model <- readr::read_rds('C:\\Users\\qlekoane\\Documents\\R Scripts\\mlOps_labs\\lab_40_docker\\lab_40_docker\\Email_Classifier\\model_xgboost.rds')


#* @apiTitle 3Sixty - New Claims - Email Classifier API
#* @apiDescription New Claims Email Query Classification.

#* Health Check - Is the API running
#* @serializer json
#* @get /health-check

status <- function(){
    list(
        status = 'API is up and running?',
        time = Sys.time()
    )
}

#* Get Predict and classify email query
#* @serializer json
#* @get /predict
function(req, res) {
    list(
        email_id = 'claim_1234567',
        email_subject = 'Please Process Claim',
        email_body = 'In Regards with Member : 04300749483, Mn: 04300749483, pr : 0437956, Sd: 2021/10/18-2021/10/19. ( Rejected: Services before dependant registration date) . member was registered 2021/01/01. please review and reprocess claim',
        locations = c('location1', 'location2')
    )
}

#* Predict and classify new email claim queries
#* @serializer json
#* @post /classify
function(req, res) {
    predict(model, new_data = as.data.frame(res$body), type = 'prob')
}


#* @plumber
#function(pr) {
 #   pr %>%
  #     pr_set_api_spec(yaml::read_yaml('openapi.yaml')) %>%
   #      pr_set_docs(docs = 'rapidoc')
#}



