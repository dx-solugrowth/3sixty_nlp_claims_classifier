
#Author: @Que
#Title: ML: Optical Character Recognition & Natural Language Processing - 3Sixty Health
#Project: NLP Classifier Script
#Date: 09 February 2023


install.packages("mlflow", dependencies = T)
devtools::install_version('h2o', version = '3.30.0.1', dependencies = TRUE)

library(mlflow)
install_mlflow()
mlflow_ui()


library(chatgpt)
library(pdftools)
library(stringr)
library(xlsx)
library(tesseract)
library(tidyverse)
library(naivebayes)

library(rpart)
library(rpart.plot)

library(gmodels)
library(e1071)
library(tidymodels)
library(textrecipes)
library(prepdat)
library(ctree)

library(pdftools)
library(tm)
library(lsa)
library(LSAfun)


# Text Mining for 3Sixty  Health - Text Exploratory Analysis ----


# Import text ----
getwd()
texts <- Corpus(DirSource('C:\\Users\\qlekoane\\Documents\\R Scripts\\lab_78_shiny_part1\\data\\classifier'))
texts <- Corpus(DirSource('C:\\Users\\qlekoane\\Documents\\R Scripts\\lab_78_shiny_part1\\data\\text'))
writeLines(as.character(texts[["6991"]]))


# Clean the text ---

rm_punc_texts <- tm_map(texts, removePunctuation)
rm_num_texts <- tm_map(rm_punc_texts, removeNumbers)
rm_punc_texts <- tm_map(rm_punc_texts, content_transformer(tolower))
rm_stops_texts <- tm_map(rm_punc_texts, removeWords, stopwords('english'))
texts_complete <- tm_map(rm_stops_texts, stripWhitespace)


# 1. Convert to a document matrix ----

texts_tdm <- DocumentTermMatrix(texts)
View(texts_corp)

texts_corp <- unlist(texts)
data_frame(line = 1:78, text = texts_corp)
texts_corp_tbl <- as_tibble(texts_corp)
view(texts_corp_tbl)

#mutate(claimLine = str_detect() ,
# claimID = str_detect(),
# priorityFlag = Str_detect('sending follow up'))
# memberNumber = Str_detect()
# tarif code =  Tariff code 97333 str_detect()
# ifelse statement for reprocessing and authorizations in one sentence
# ifelse claim 45856066 ID then flag column to classify as query
# ifelse has HSD with auth no then make unknown
# ifelse reprocessing and medication then unknown

texts_clean <- texts_corp_tbl %>%

    mutate(queryNumber = row_number()) %>%

    mutate(
        claims = case_when(

            # Reprocessing Claims Signals ----
            str_detect(value, 'reprocess') ~ 'reprocess_claim',
            str_detect(value, 'reprocessing this claim modfifier') ~ 'reprocess_claim',
            str_detect(value, 'review of line') ~ 'reversal_correct',

            # New Claims Signals ----
            str_detect(value, 'short paid') ~ 'new_claim',
            str_detect(value, 'claim not on system') ~ 'new_claim',
            str_detect(value, 'review of claim') ~ 'new_claim',
            str_detect(value, 'account for processing') ~ 'new_claim',
            str_detect(value, 'paper claim') ~ 'new_claim',
            str_detect(value, 'refund back') ~ 'new_claim',

            # Query Authorisations Signals ----
            str_detect(value, 'approve authorisation urgently to include') ~ 'query_authorisation',
            str_detect(value, 'claim not linking to the authorisation') ~ 'query_authorisation',
            str_detect(value, 'reprocessed') ~ 'query_authorisation',
            str_detect(value, 'auth') ~ 'query_authorisation',
            str_detect(value, 'Hospital') ~ 'query_authorisation',

            str_detect(value, 'Invalid/incorrect shortpayment,') ~ 'query',
            str_detect(value, 'status') ~ 'query',

            # Edge Cases Signals ----
            TRUE ~ 'unknown'
        )
    ) %>%

    mutate(
        priority =
        case_when(
            str_detect(value, 'sending follow up') ~ 'medium',
            str_detect(value, 'urgently') ~ 'high',
            TRUE ~ 'low'
        )
    ) %>%
    mutate(
        authorisation =
            case_when(
                str_detect(value, 'review of line') ~ 1 %>% as.integer(),
                str_detect(value, 'neo') ~ 1 %>% as.integer(),
                TRUE ~ 0 %>% as.integer()
            )
    ) %>%
    mutate(
        reversal_correct =
            case_when(
                str_detect(value, 'auth') ~ 1 %>% as.integer(),
                str_detect(value, 'authorisation') ~ 1 %>% as.integer(),
                TRUE ~ 0 %>% as.integer()
            )
    ) %>%
    mutate(
        reprocess_claim =
            case_when(
                str_detect(value, 'reprocess') ~ 1 %>% as.integer(),
                str_detect(value, 'reprocessing this claim modfifier') ~ 1 %>% as.integer(),
                TRUE ~ 0 %>% as.integer()
            )
    ) %>%
    mutate(
        new_claim =
            case_when(
                str_detect(value, 'short paid') ~ 1 %>% as.integer(),
                str_detect(value, 'claim not on system') ~ 1 %>% as.integer(),
                str_detect(value, 'review of claim') ~ 1 %>% as.integer(),
                str_detect(value, 'account for processing') ~ 1 %>% as.integer(),
                str_detect(value, 'paper claim') ~ 1 %>% as.integer(),
                str_detect(value, 'refund back') ~ 1 %>% as.integer(),
                TRUE ~ 0 %>% as.integer()
            )
    ) %>%
    mutate(
        query_authorisation =
            case_when(
                str_detect(value, 'approve authorisation urgently to include') ~ 1 %>% as.integer(),
                str_detect(value, 'claim not linking to the authorisation') ~ 1 %>% as.integer(),
                str_detect(value, 'reprocessed') ~ 1 %>% as.integer(),
                str_detect(value, 'auth') ~ 1 %>% as.integer(),
                str_detect(value, 'Hospital') ~ 1 %>% as.integer(),
                TRUE ~ 0 %>% as.integer()
            )
    ) %>%
    mutate(
        query =
            case_when(
                str_detect(value, 'Invalid/incorrect shortpayment,') ~ 1 %>% as.integer(),
                str_detect(value, 'status') ~ 1 %>% as.integer(),
                TRUE ~ 0 %>% as.integer()
            )
    ) %>%
    mutate(
        unknown =
            case_when(
                str_detect(claims, 'unknown') ~ 1 %>% as.integer(),
                TRUE ~ 0 %>% as.integer()
            )
    ) %>% view()


# final classification rules for the decision tree classifier algorithm ----

texts_clean <- texts_clean %>%
    mutate(
        type = if(reversal_correct == 1 && query_authorisation == 1){'query_authoridation'}
            ) %>% view()

# 1.1 ---- Decision Tree Model Build -----

set.seed(2023)
dt_text_clean <- texts_clean %>% select(-value, -queryNumber, -priority)
glimpse(dt_text_clean)

index_set <- sample(2, nrow(dt_text_clean), replace = TRUE, prob = c(0.7, 0.3))

train <- dt_text_clean[index_set==1, ]
test <- dt_text_clean[index_set==2, ]

decisionTree_results <- rpart(claims ~authorisation + reversal_correct + reprocess_claim +
                                  new_claim + query_authorisation + query,
                              method = 'class',
                              data = train )

rpart.plot(decisionTree_results, extra = 5, cex = .4)


write.csv(nlp_predictions, "C:\\Users\\qlekoane\\Documents\\nlp_predictions_v2.csv")

nlp_predictions <- bind_cols(dt_text_clean, texts_corp_tbl)
view(nlp_predictions)


# 2.  lexicon build ----

custom_lexicon <- structure(list(
    word = c('claim', 'reprocessing', 'reprocess'),
    class = c('query', 'claim')
))


texts_clean %>%
    unnest_tokens(word, text) %>%
    inner_join(custom_lexicon)


# 3. Sentence structure analysis ----

text <- 'Good day, I have submitted a claim for refund back to me on 14/03/2022. To date I have not received any responses to my request. I have been sending follow up emails and still nothing. I also aksed for a new medical aid car whereby i received this ref - WF20220314N8048978 I dont want a piece of paper with the new medical aid details on, I want a card to be sent to me. AGAIN, find attached my documents for my claim. I have already settlemed this with the doctor and would like my refund to be paid back to me. Can you please contact me so that I can have some confirmation that you have received this email and are attending to my claim. Regards, A Smith'


# 4. plot dependency parser results -----

tx <- udpipe(texts, 'english')
textplot_dependencyparser(tx)



# 5. Build ML Classification Algorithm - Use Naive Bayes Classification Model -----

str(texts_corp_tbl)
table(texts_corp_tbl$value)

train_labels <- texts_corp_tbl[]
test_labels
texts_corp_tbl



cols <- 10 ; rows <- 100 ; probs <- c("0" = 0.9, "1" = 0.1)
M <- matrix(sample(0:1, rows * cols, TRUE, probs), nrow = rows, ncol = cols)
y <- factor(sample(paste0("class", LETTERS[1:2]), rows, TRUE, prob = c(0.3,0.7)))
colnames(M) <- paste0("V", seq_len(ncol(M)))
laplace <- 0

M <-


### Train the Bernoulli Naive Bayes
bnb <- bernoulli_naive_bayes(x = M, y = y, laplace = laplace)
summary(bnb)

# Classification
head(predict(bnb, newdata = M, type = "class")) # head(bnb %class% M)

# Posterior probabilities
head(predict(bnb, newdata = M, type = "prob")) # head(bnb %prob% M)

# Parameter estimates
coef(bnb)


# 6. UAT -  Model

# Import the prostate dataset into H2O:
prostate <- h2o.importFile("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate.csv")

# Set the predictors and response; set the response as a factor:
prostate$CAPSULE <- as.factor(prostate$CAPSULE)
predictors <- c("ID", "AGE", "RACE", "DPROS" ,"DCAPS" ,"PSA", "VOL", "GLEASON")
response <- "CAPSULE"

# Build and train the model:
pros_nb <- h2o.naiveBayes(x = predictors,
                          y = response,
                          training_frame = prostate,
                          laplace = 0,
                          nfolds = 5,
                          seed = 1234)

# Eval performance:
perf <- h2o.performance(pros_nb)

# Generate the predictions on a test set (if necessary):
pred <- h2o.predict(pros_nb, newdata = prostate)





# Text Feature Encoding Automated Process-----

recipe_spec_tokens <- recipe(~., data = texts_corp_tbl, select(value)) %>%
    step_mutate(description = value) %>%
    step_tokenize(description) %>%
    #step_stem(description)

    recipe_spec_tokens %>% recipes::prep() %>% juice() %>% glimpse()

first_tokenize <- recipe_spec_tokens %>%
    recipes::prep() %>%
    juice() %>%
    slice(1) %>%
    pull(description)

unclass(first_tokenize)

text_model_features <- recipe(~., data = texts_corp_tbl %>% select(value)) %>%

    step_mutate(ngram = value) %>%
    step_tokenize(ngram) %>%
    step_stem(ngram) %>%
    step_stopwords(ngram) %>%
    step_rm(contains(c('description', 'six', 'address', 'regards', '3sixtyhealth.co.za'))) %>%

    step_ngram(ngram, num_tokens = 2, min_num_tokens = 1) %>%
    step_tokenfilter(ngram, max_tokens = 300) %>%

    step_tf(ngram, weight_scheme = 'binary') %>%
    step_mutate_at(starts_with('tf_', fn = as.integer))


text_model_features %>% recipes::prep() %>% juice() %>% view()
