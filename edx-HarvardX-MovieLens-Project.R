################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# following libraries will be used 
library(ggplot2)
library(gridExtra)
library(tidyverse)
library(caret)
library(lubridate)
library(data.table)
library(graphics)
library(Matrix)
library(irlba)
library(recosystem)


# basic info of the two sets

knitr::kable(head(edx), caption = "Sample data from edx")
data_dim <- rbind(dim(edx),dim(validation))
colnames(data_dim) <- c("no of record", "no of variable")
rownames(data_dim) <- c("edx", "validation")
knitr::kable(data_dim, caption = "Dimensions of edx and validation")

# save as Rda files for future use under working directory

save(edx, file = "edx.Rda")
save(validation, file = "validation.Rda")

# further tidy the data as below, split the movie title and release year, change timestamp as date

load("edx.Rda")
load("validation.Rda")

# clean edx and saved as training_set
training_set <- edx %>% 
  mutate(rating_date=date(as_datetime(timestamp)),
         release_year=str_remove_all(str_extract(title, "\\(\\d{4}\\)"),"\\(|\\)"),
         tidy_title=trimws(str_remove_all(title, "\\(\\d{4}\\)"))) %>% 
  dplyr::select(userId, movieId, rating, rating_date, release_year, tidy_title, genres)

# clean validation and saved as test_set
test_set <- validation %>% 
  mutate(rating_date=date(as_datetime(timestamp)),
         release_year=str_remove_all(str_extract(title, "\\(\\d{4}\\)"),"\\(|\\)"),
         tidy_title=trimws(str_remove_all(title, "\\(\\d{4}\\)"))) %>% 
  dplyr::select(userId, movieId, rating, rating_date, release_year, tidy_title, genres)

# save as Rda files for future use under working directory
save(training_set, file = "training_set.Rda")
save(test_set, file = "test_set.Rda")

# load the cleaned data sets to start analysis 
load("training_set.Rda")
load("test_set.Rda")

# have a look and for report 
knitr::kable(head(training_set), caption = "Sample data from training_set")

# EDA (exploratory data analysis) rating distribution 
training_set %>% 
  ggplot(aes(rating)) + 
  geom_histogram(fill = "#56B4E9") + 
  ggtitle("Chart 1: Rating Distribution") + 
  ylab("no of ratings")

p1 <- training_set %>% 
  group_by(movieId) %>% 
  summarise(movie_avg_rating = mean(rating)) %>%
  ggplot(aes(movie_avg_rating)) + 
  geom_histogram(bins = 50, fill = "#56B4E9") + 
  ggtitle("Chart 2: Movie Avg Rating Distribution") + 
  ylab("no of movies")

p2 <- training_set %>% 
  group_by(movieId) %>% 
  summarise(movie_avg_rating = mean(rating)) %>%
  ggplot(aes(sample = movie_avg_rating)) + 
  geom_qq() + geom_qq_line() + 
  ggtitle("Chart 3: Movie Avg Rating QQ plot") 

p3 <- training_set %>% 
  group_by(userId) %>% 
  summarise(user_avg_rating = mean(rating)) %>%
  ggplot(aes(user_avg_rating)) + 
  geom_histogram(bins = 50, fill = "#56B4E9") + 
  ggtitle("Chart 4: User Avg Rating Distribution") + 
  ylab("no of users")

p4 <- training_set %>% 
  group_by(userId) %>% 
  summarise(user_avg_rating = mean(rating)) %>%
  ggplot(aes(sample = user_avg_rating)) + 
  geom_qq() + geom_qq_line() + 
  ggtitle("Chart 5: User Avg Rating QQ plot") 

grid.arrange(p1, p2, p3, p4, nrow=2)
rm(p1, p2, p3, p4)

# summary of basic statistics
rating <- summary(training_set$rating)

movie_avg_rating <- summary(training_set %>% 
                              group_by(movieId) %>% 
                              summarise(movie_avg_rating = mean(rating)) %>% 
                              pull(movie_avg_rating))

user_avg_rating <- summary(training_set %>% 
                             group_by(userId) %>% 
                             summarise(user_avg_rating = mean(rating)) %>% 
                             pull(user_avg_rating))

knitr::kable(rbind(rating, movie_avg_rating, user_avg_rating),
             caption = "Summary of Statistics")
rm(rating, user_avg_rating, movie_avg_rating)

# build RMSE function for cross validation & evaluation
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# naive model
mu_hat <- mean(training_set$rating)
print(paste0("Overall average rating is ", mu_hat))
naive_rmse <- RMSE(test_set$rating, rep(mu_hat, length(test_set$rating)))
print(paste0("RMSE is ", naive_rmse))

# rain_set and cv_set are created from training_set
set.seed(1)
cv_index <- createDataPartition(y = training_set$rating, times = 1, p = 0.1, list = FALSE)
train_set <- training_set[-cv_index,]
temp <- training_set[cv_index,]

# make sure userId and movieId in cv_set are also in train_set
cv_set <- temp %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

# add rows removed from cv_set back into train_set
removed <- anti_join(temp, cv_set)
train_set <- rbind(train_set, removed)
rm(cv_index, removed, temp, naive_rmse)

# tunable regularization parameter lambda 
lambda_b_tune <- seq(4, 6, 0.05)

# find the minimum rmse 
rmse_baseline_reg <- sapply(lambda_b_tune, function(x) {
  # tune movie bias
  b_i_reg <- train_set %>% 
    group_by(movieId) %>% 
    summarise(movie_bias_reg = sum(rating - mu_hat)/(x + n()))
  
  # tune user bias
  b_u_reg <- train_set %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    group_by(userId) %>% 
    summarise(user_bias_reg = sum(rating - mu_hat- movie_bias_reg)/(x + n()))
  
  # cross-validation with cv_set
  rating_hat <- cv_set %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    left_join(b_u_reg, by = "userId") %>%
    mutate(rating_hat=movie_bias_reg + user_bias_reg + mu_hat) %>% 
    pull(rating_hat)
  RMSE(cv_set$rating, rating_hat)
})

# plot lambda vs rmse_baseline_reg
save(lambda_b_tune, file = "lambda_b_tune.Rda")
save(rmse_baseline_reg, file = "rmse_baseline_reg.Rda")
qplot(lambda_b_tune, rmse_baseline_reg, main = "Chart 6: Lambda vs RMSE")

# best lambda that minimizes rmse
lambda_b <- lambda_b_tune[which.min(rmse_baseline_reg)]

# regularized estimates of movie bias
b_i_reg <- training_set %>% group_by(movieId) %>% 
  summarise(movie_bias_reg = sum(rating - mu_hat)/(lambda_b + n()))

# regularized estimates of user bias
b_u_reg <- training_set %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  group_by(userId) %>% 
  summarise(user_bias_reg = sum(rating - mu_hat- movie_bias_reg)/(lambda_b + n()))

# validate with test_set
rating_predicted <- test_set %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  mutate(pred = mu_hat + movie_bias_reg + user_bias_reg) %>%
  pull(pred)

rmse_baseline <- RMSE(test_set$rating, rating_predicted)
save(rmse_baseline, file = "rmse_baseline.Rda")
print(paste0("RMSE for baseline predictor model is ", rmse_baseline))

# caution, there are unreasonable predictions to be modified, such as
unreasonable <- cbind(test_set[c(which.min(rating_predicted),which.max(rating_predicted)),], 
      rating_predicted = c(min(rating_predicted), max(rating_predicted))) %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  mutate(global_mean = mu_hat) %>%
  dplyr::select(-genres, -rating_date, -release_year) 

save(unreasonable, file = "unreasonable.Rda")
knitr::kable(unreasonable, caption = "Unreasonable Predictions")

rm(rmse_baseline_reg, rmse_baseline, rating_predicted, b_i_reg, b_u_reg)
# known highest rating is 5, and lowest is 0.5, so adjust predtions based on this info going forward
# so we have to train the data again with adjusted cross-validation to get bset lambda
rmse_baseline_reg_adj <- sapply(lambda_b_tune, function(x) {
  # tune movie bias
  b_i_reg <- train_set %>% 
    group_by(movieId) %>% 
    summarise(movie_bias_reg = sum(rating - mu_hat)/(x + n()))
  
  # tune user bias
  b_u_reg <- train_set %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    group_by(userId) %>% 
    summarise(user_bias_reg = sum(rating - mu_hat- movie_bias_reg)/(x + n()))
  
  # cross-validation with cv_set
  rating_hat <- cv_set %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    left_join(b_u_reg, by = "userId") %>%
    mutate(temp_pred=movie_bias_reg + user_bias_reg + mu_hat) %>% 
    mutate(rating_hat = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
    pull(rating_hat)
  RMSE(cv_set$rating, rating_hat)
})

# plot lambda vs rmse_baseline_reg_adj
save(rmse_baseline_reg_adj, file = "rmse_baseline_reg_adj.Rda")
qplot(lambda_b_tune, rmse_baseline_reg_adj, main = "Chart 7: Lambda vs RMSE")

# best lambda that minimizes rmse
lambda_b <- lambda_b_tune[which.min(rmse_baseline_reg_adj)]

# regularized estimates of movie bias
b_i_reg <- training_set %>% group_by(movieId) %>% 
  summarise(movie_bias_reg = sum(rating - mu_hat)/(lambda_b + n()))

# regularized estimates of user bias
b_u_reg <- training_set %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  group_by(userId) %>% 
  summarise(user_bias_reg = sum(rating - mu_hat- movie_bias_reg)/(lambda_b + n()))

# validate with test_set again by adjusting unreasonable predictions
rating_predicted <- test_set %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  mutate(pred = mu_hat + movie_bias_reg + user_bias_reg) %>% 
  mutate(pred_adj = ifelse(pred < 0.5, 0.5, ifelse(pred > 5, 5, pred))) 

rmse_baseline_adj <- RMSE(test_set$rating, rating_predicted$pred_adj)
save(rmse_baseline_adj, file = "rmse_baseline_adj.Rda")
print(paste0("RMSE for baseline predictor model is ", rmse_baseline_adj))

# explore the residuals from baseline predictors model
training_temp <- training_set %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>% 
  mutate(temp_pred = mu_hat + movie_bias_reg + user_bias_reg) %>% 
  mutate(baseline_pred = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
  dplyr::select(-movie_bias_reg, -user_bias_reg, -temp_pred)
save(training_temp, file = "training_temp.Rda")

# residual by release year has a declining trend
p1 <- training_temp %>% 
  group_by(release_year) %>% 
  summarise(avg_res=mean(rating - baseline_pred)) %>% 
  ggplot(aes(release_year, avg_res)) + 
  geom_point() + 
  theme(axis.text.x = element_text(angle = 90, size = 5)) + 
  scale_y_continuous(limits = c(-0.06, 0.25)) + 
  ggtitle("Chart 8: Avg Residual For Each Release Year")

# residual by rating month has a slightly declining trend too
p2 <- training_temp %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  group_by(rating_month) %>% 
  summarise(avg_res=mean(rating - baseline_pred)) %>% 
  ggplot(aes(rating_month, avg_res)) + 
  geom_point() + 
  theme(axis.text.x = element_text(angle = 90)) + 
  scale_y_continuous(limits = c(-0.06, 0.25)) + 
  ggtitle("Chart 9: Avg Residual For Each Rating Month")

grid.arrange(p1, p2, nrow=1)
rm(p1, p2)

# still use regularization to get release year bias and rating month bias with train_set
# cross-validate with cv_set 
temp_train <- train_set %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>% 
  mutate(temp_pred = mu_hat + movie_bias_reg + user_bias_reg) %>% 
  mutate(baseline_pred = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
  dplyr::select(-movie_bias_reg, -user_bias_reg, -temp_pred)

# tunable regularization parameter lambda 
lambda_t_tune <- seq(12, 14, 0.05)

# find the minimum rmse 
rmse_b_t_reg_tune <- sapply(lambda_t_tune, function(x) {
  b_releasey_reg <- temp_train %>% 
    group_by(release_year) %>%
    summarise(releasey_bias_reg = sum(rating - baseline_pred)/(x+n())) 
  
  b_rm_reg <- temp_train %>% 
    mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
    left_join(b_releasey_reg, by = "release_year") %>% 
    group_by(rating_month) %>% 
    summarise(rm_bias_reg = sum(rating - baseline_pred - releasey_bias_reg)/(x+n()))
  
  rating_hat <- cv_set %>% 
    mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    left_join(b_u_reg, by = "userId") %>% 
    left_join(b_releasey_reg, by = "release_year") %>% 
    left_join(b_rm_reg, by = "rating_month") %>% 
    mutate(temp_pred=movie_bias_reg + user_bias_reg + mu_hat + releasey_bias_reg + rm_bias_reg) %>% 
    mutate(rating_hat = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
    pull(rating_hat)
  
  RMSE(cv_set$rating, rating_hat)
})

save(lambda_t_tune, file = "lambda_t_tune.Rda")
save(rmse_b_t_reg_tune, file = "rmse_b_t_reg_tune.Rda")
qplot(lambda_t_tune, rmse_b_t_reg_tune, main = "Chart 10: Lambda vs RMSE")

# best lambda that minimizes rmse
lambda_t <- lambda_t_tune[which.min(rmse_b_t_reg_tune)]

# regularized estimates of release year bias
b_releasey_reg <- training_temp %>% 
  group_by(release_year) %>%
  summarise(releasey_bias_reg = sum(rating - baseline_pred)/(lambda_t + n())) 

# regularized estimates of rating month  bias
b_rm_reg <- training_temp %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  left_join(b_releasey_reg, by = "release_year") %>% 
  group_by(rating_month) %>% 
  summarise(rm_bias_reg = sum(rating - baseline_pred - releasey_bias_reg)/(lambda_t + n()))

# validate with test_set
rating_predicted <- test_set %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_releasey_reg, by = "release_year") %>% 
  left_join(b_rm_reg, by = "rating_month") %>% 
  mutate(temp_pred = mu_hat + movie_bias_reg + user_bias_reg + releasey_bias_reg + rm_bias_reg) %>% 
  mutate(pred = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
  pull(pred)

rmse_b_t_reg <- RMSE(test_set$rating, rating_predicted)
save(rmse_b_t_reg, file = "rmse_b_t_reg.Rda")
print(paste0("RMSE for baseline + time predictor model is ", rmse_b_t_reg))

# let's see if RMSE decrease, if we regularize these four predictors symmetrically.
# tunable regularization parameter lambda
lambda_tune <- seq(4, 6, 0.05)

# find the minimum rmse with symmetrical regularization, taking about 10 mins to finish 
rmse_reg <- sapply(lambda_tune, function(x) {
  b_i_reg <- train_set %>% 
    group_by(movieId) %>% 
    summarise(movie_bias_reg = sum(rating - mu_hat)/(x + n()))
  
  b_u_reg <- train_set %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    group_by(userId) %>% 
    summarise(user_bias_reg = sum(rating - mu_hat- movie_bias_reg)/(x + n()))
  
  b_releasey_reg <- train_set %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    left_join(b_u_reg, by = "userId") %>% 
    group_by(release_year) %>%
    summarise(releasey_bias_reg = sum(rating - mu_hat- movie_bias_reg - user_bias_reg)/(x + n())) 
  
  b_rm_reg <- train_set %>% 
    mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    left_join(b_u_reg, by = "userId") %>% 
    left_join(b_releasey_reg, by = "release_year") %>% 
    group_by(rating_month) %>% 
    summarise(rm_bias_reg = sum(rating - mu_hat- movie_bias_reg - user_bias_reg - 
                                  releasey_bias_reg)/(x + n()))
  
  rating_hat <- cv_set %>% 
    mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
    left_join(b_i_reg, by = "movieId") %>% 
    left_join(b_u_reg, by = "userId") %>% 
    left_join(b_releasey_reg, by = "release_year") %>% 
    left_join(b_rm_reg, by = "rating_month") %>% 
    mutate(temp_pred = movie_bias_reg + user_bias_reg + mu_hat + releasey_bias_reg + rm_bias_reg) %>% 
    mutate(rating_hat = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
    pull(rating_hat)
  
  RMSE(cv_set$rating, rating_hat)
})

save(lambda_tune, file = "lambda_tune.Rda")
save(rmse_reg, file = "rmse_reg.Rda")
qplot(lambda_tune, rmse_reg, main = "Chart 11: Lambda vs RMSE")

# best lambda that minimizes rmse
lambda <- lambda_tune[which.min(rmse_reg)]

# regularized estimates of movie bias
b_i_reg <- training_set %>% 
  group_by(movieId) %>% 
  summarise(movie_bias_reg = sum(rating - mu_hat)/(lambda + n()))

# regularized estimates of user bias
b_u_reg <- training_set %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  group_by(userId) %>% 
  summarise(user_bias_reg = sum(rating - mu_hat- movie_bias_reg)/(lambda + n()))

# regularized estimates of release year bias
b_releasey_reg <- train_set %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  group_by(release_year) %>%
  summarise(releasey_bias_reg = sum(rating - mu_hat- movie_bias_reg - user_bias_reg)/(lambda + n())) 

# regularized estimates of rating month  bias
b_rm_reg <- train_set %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_releasey_reg, by = "release_year") %>% 
  group_by(rating_month) %>% 
  summarise(rm_bias_reg = sum(rating - mu_hat- movie_bias_reg - user_bias_reg - releasey_bias_reg)/(lambda + n()))

# validate with test_set
rating_predicted <- test_set %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_releasey_reg, by = "release_year") %>% 
  left_join(b_rm_reg, by = "rating_month") %>% 
  mutate(pred = mu_hat + movie_bias_reg + user_bias_reg + releasey_bias_reg + rm_bias_reg) %>% 
  mutate(pred_adj = ifelse(pred < 0.5, 0.5, ifelse(pred > 5, 5, pred))) %>% 
  pull(pred_adj)

rmse_b_t_reg_sym <- RMSE(test_set$rating, rating_predicted)
save(rmse_b_t_reg_sym, file = "rmse_b_t_reg_sym.Rda")
print(paste0("RMSE for baseline + time predictor model is ", rmse_b_t_reg_sym))

# modify train_set, sv_set, and training_set
train_set_modified <- train_set %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_releasey_reg, by = "release_year") %>% 
  left_join(b_rm_reg, by = "rating_month") %>% 
  mutate(temp_pred = mu_hat + movie_bias_reg + user_bias_reg + releasey_bias_reg + rm_bias_reg) %>% 
  mutate(b_t_pred = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
  dplyr::select(-movie_bias_reg, -user_bias_reg, -releasey_bias_reg, -rm_bias_reg, -temp_pred)

cv_set_modified <- cv_set %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_releasey_reg, by = "release_year") %>% 
  left_join(b_rm_reg, by = "rating_month") %>% 
  mutate(temp_pred = mu_hat + movie_bias_reg + user_bias_reg + releasey_bias_reg + rm_bias_reg) %>% 
  mutate(b_t_pred = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
  dplyr::select(-movie_bias_reg, -user_bias_reg, -releasey_bias_reg, -rm_bias_reg, -temp_pred)

training_set_modified <- training_set %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_releasey_reg, by = "release_year") %>% 
  left_join(b_rm_reg, by = "rating_month") %>% 
  mutate(temp_pred = mu_hat + movie_bias_reg + user_bias_reg + releasey_bias_reg + rm_bias_reg) %>% 
  mutate(b_t_pred = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
  dplyr::select(-movie_bias_reg, -user_bias_reg, -releasey_bias_reg, -rm_bias_reg, -temp_pred)

rm(temp_train, training_temp, lambda, lambda_tune, rmse_reg, rating_predicted, train_set, cv_set, training_set) 
save(train_set_modified, file = "train_set_modified.Rda")
save(cv_set_modified, file = "cv_set_modified.Rda")
save(training_set_modified, file = "training_set_modified.Rda")

# take popular movies and active users from traning_set 
corr_check <- training_set_modified %>% group_by(movieId) %>%
  filter(n() >= 2000) %>% ungroup() %>% group_by(userId) %>% 
  filter(n() >= 300) %>% ungroup()

# image function similar to the one provided during the Machine Learning course 
my_image <- function(x, zlim = range(x), str_title, ...){
  colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))
  cols <- 1:ncol(x)
  rows <- 1:nrow(x)
  image(rows, cols, x, xaxt = "n", yaxt = "n",
        xlab="", ylab="",  col = colors, zlim = zlim, main = str_title, ...)
  axis(side = 2, cols, colnames(x), las = 2,  cex.axis=0.7)
  legend("topright", inset=c(-0.15, 0.05), legend = round(seq(min(x), max(x), length.out = 9), 1), 
         col = colors, lty = rep(1,9), lwd = rep(5,9), seg.len = 0.06, text.width = rep(0.05,9), 
         xpd = TRUE, cex = 0.8, bty = "n", xjust = 0, yjust = 0, y.intersp=0.7, x.intersp = 0.05)
}

# set graphical parameters for plots, and save for reprot
png(filename="Image.png", bg = "transparent", width = 1000, height = 869)
op <- par(mar = c(2,15,2,6) + 0.1, mfrow = c(2, 1))
# interactions between users and movies via residuals 
matrix_temp_01 <- corr_check %>% filter(tidy_title == "Shawshank Redemption, The" | 
                                      tidy_title == "Schindler's List" |
                                      tidy_title %like% "Lord of the Rings:" | 
                                      tidy_title == "Titanic") %>% 
  mutate(res = rating - b_t_pred) %>% 
  arrange(tidy_title) %>% 
  dplyr::select(userId, tidy_title, res) %>% 
  spread(tidy_title, res) %>% 
  arrange(`Lord of the Rings: The Fellowship of the Ring, The`) %>% 
  drop_na() %>% 
  as.matrix()

# interactions between users and movies via ratings
matrix_temp_02 <- corr_check %>% filter(tidy_title == "Shawshank Redemption, The" | 
                                      tidy_title == "Schindler's List" | 
                                      tidy_title %like% "Lord of the Rings:" | 
                                      tidy_title == "Titanic") %>% 
  arrange(tidy_title) %>% 
  dplyr::select(userId, tidy_title, rating) %>% 
  spread(tidy_title, rating) %>% 
  arrange(`Lord of the Rings: The Fellowship of the Ring, The`) %>% 
  drop_na() %>% 
  as.matrix()

# plot Residual Image for Interactions between Users and Movies
matrix_temp_01[is.na(matrix_temp_01)] <- 0
rownames(matrix_temp_01)<- matrix_temp_01[,1]
matrix_temp_01 <- matrix_temp_01[,-1]

my_image(matrix_temp_01, str_title = "Chart 12: Residual Image for Interactions between Users and Movies")

# plot Actual Rating Image for Interactions between Users and Movies
matrix_temp_02[is.na(matrix_temp_02)] <- 0
rownames(matrix_temp_02)<- matrix_temp_02[,1]
matrix_temp_02 <- matrix_temp_02[,-1]

my_image(matrix_temp_02, str_title = "Chart 13: Actual Rating Image for Interactions between Users and Movies")

# reset graphical parameters and remove temporary data frame
par(op)
dev.off()
corr_check %>% filter(tidy_title == "Shawshank Redemption, The" | 
                        tidy_title == "Schindler's List" | 
                        tidy_title %like% "Lord of the Rings:" | 
                        tidy_title == "Titanic") %>% 
  arrange(tidy_title) %>% 
  dplyr::select(userId, tidy_title, rating) %>% 
  filter(userId %in% c(70078, 24544)) %>% 
  spread(userId, rating) %>% 
  knitr::kable(caption = "Interactions between Users and Movies")

rm(corr_check, df_temp_01, df_temp_02)

# SVD is used for addressing interactions between users and movies
# use irlba package to do SVD for sparse matrix 
train_interaction <- train_set_modified %>% 
  mutate(res = rating - b_t_pred, 
         movie_index = as.numeric(factor(movieId)), 
         user_index = as.numeric(factor(userId))) %>% 
  dplyr::select(userId, movieId, movie_index, user_index, res)

interaction_sparse <- sparseMatrix(i = train_interaction$user_index, 
                                   j = train_interaction$movie_index, 
                                   x = train_interaction$res,
                                   dimnames = list(levels(factor(train_interaction$userId)), 
                                                   levels(factor(train_interaction$movieId))))

# tunable parameter is the number of factors f, validate with test_set to find the best one, taking about 20 mins
cv_set_modified <- cv_set_modified %>%  mutate(res = rating - b_t_pred) 
f <- c(50, 55, 60)
svd_rmse <- sapply(f, function(f) {
  interaction_svd <- irlba(interaction_sparse, nv=f)
  rownames(interaction_svd$u) <- rownames(interaction_sparse)
  rownames(interaction_svd$v) <- colnames(interaction_sparse)
  
  k <- seq(1:dim(cv_set_modified)[1])
  res_pred <- sapply(k, function(k) {
    i <- as.character(cv_set_modified$userId[k])
    j <- as.character(cv_set_modified$movieId[k])
    sum(interaction_svd$u[i,] * interaction_svd$d * interaction_svd$v[j,])
  })
  
  RMSE(cv_set_modified$res, res_pred)
})

save(f, file = "f.Rda")
save(svd_rmse, file = "svd_rmse.Rda")
qplot(f, svd_rmse, main = "Chart 14: Number of Features vs RMSE")
f <- f[which.min(svd_rmse)]

interaction_svd <- irlba(interaction_sparse, nv=f)
rownames(interaction_svd$u) <- rownames(interaction_sparse)
rownames(interaction_svd$v) <- colnames(interaction_sparse)

# modify training_set, optional, taking about 1 hour 
k <- seq(1:dim(training_set_modified)[1])
res_pred <- sapply(k, function(k) {
  i <- as.character(training_set_modified$userId[k])
  j <- as.character(training_set_modified$movieId[k])
  sum(interaction_svd$u[i,] * interaction_svd$d * interaction_svd$v[j,])
})

training_set_pred <- cbind(training_set_modified, res_pred = res_pred) %>% 
  mutate(temp_pred = b_t_pred + res_pred) %>% 
  mutate(final_pred = ifelse(temp_pred < 0.5, 0.5, ifelse(temp_pred > 5, 5, temp_pred))) %>% 
  dplyr::select(-temp_pred)

save(training_set_pred, file = "training_set_pred.Rda")
RMSE(training_set_pred$rating, training_set_pred$final_pred)
rm(k, res_pred, train_interaction, interaction_sparse)

# interactions between users and movies via residuals from the final model
matrix_temp_01 <- training_set_pred %>% group_by(movieId) %>%
  filter(n() >= 2000) %>% ungroup() %>% group_by(userId) %>% 
  filter(n() >= 300) %>% ungroup() %>% 
  filter(tidy_title == "Shawshank Redemption, The" | 
           tidy_title == "Schindler's List" | 
           tidy_title %like% "Lord of the Rings:" | 
           tidy_title == "Titanic") %>% 
  mutate(res = rating - final_pred) %>% 
  arrange(tidy_title) %>% 
  dplyr::select(userId, tidy_title, res) %>% 
  spread(tidy_title, res) %>% 
  arrange(`Lord of the Rings: The Fellowship of the Ring, The`) %>% 
  drop_na() %>% 
  as.matrix()

# plot Residual Image for Interactions between Users and Movies
png(filename="Final_res.png", bg = "transparent", width = 1000, height = 869)
op <- par(mar = c(2,15,2,6) + 0.1, mfrow = c(2, 1))
matrix_temp_01[is.na(matrix_temp_01)] <- 0
rownames(matrix_temp_01)<- matrix_temp_01[,1]
matrix_temp_01 <- matrix_temp_01[,-1]

my_image(matrix_temp_01, str_title = "Chart 15: Final Residual Image for Interactions between Users and Movies")

par(op)
dev.off()
rm(matrix_temp_01)

# validate with test_set
k <- seq(1:dim(test_set)[1])
res_pred_svd <- sapply(k, function(k) {
  i <- as.character(test_set$userId[k])
  j <- as.character(test_set$movieId[k])
  sum(interaction_svd$u[i,] * interaction_svd$d * interaction_svd$v[j,])
})
test_set <- cbind(test_set, res_pred_svd)
rating_predicted <- test_set %>% 
  mutate(rating_month = round_date(rating_date, unit = "month")) %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_releasey_reg, by = "release_year") %>% 
  left_join(b_rm_reg, by = "rating_month") %>% 
  mutate(pred = mu_hat + movie_bias_reg + user_bias_reg + releasey_bias_reg + rm_bias_reg + res_pred_svd) %>% 
  mutate(pred_adj = ifelse(pred < 0.5, 0.5, ifelse(pred > 5, 5, pred))) %>% 
  pull(pred_adj)

rmse_b_t_reg_svd <- RMSE(test_set$rating, rating_predicted)
save(rmse_b_t_reg_svd, file = "rmse_b_t_reg_svd.Rda")
print(paste0("RMSE for baseline + time predictor + svd model is ", rmse_b_t_reg_svd))

# remove intermediate data files
file.remove("cv_set_modified.Rda", "lambda_b_tune.Rda", "lambda_t_tune.Rda", "lambda_tune.Rda", "rmse_b_t_reg.Rda", 
            "rmse_b_t_reg_sym.Rda", "rmse_baseline.Rda", "rmse_baseline_adj.Rda", "rmse_baseline_reg.Rda", 
            "rmse_baseline_reg_adj.Rda", "rmse_reg.Rda", "train_set_modified.Rda", "training_set_modified.Rda", 
            "training_temp.Rda", "unreasonable.Rda", "f.Rda", "rmse_b_t_reg_svd.Rda", "rmse_b_t_reg_tune.Rda", 
            "svd_rmse.Rda", "training_set_pred.Rda")
#-----------------------------------------------------------------------------------------------
# predicting with the help of recosystem package, taking about 1 hour
training_data <- training_set %>% 
  dplyr::select(userId, movieId, rating) %>% 
  write.table(file = "training_data.txt", sep = " ", row.names = FALSE, col.names = FALSE)
training_data <- data_file(file.path(getwd(), "training_data.txt"), index1 = TRUE)
r = Reco()
set.seed(1)
opts_tune <- r$tune(training_data, opts = list(dim = c(10, 20),costp_l2 = 0.01, costq_l2 = 0.01))
r$train(training_data, 
        out_model = file.path(getwd(), "model_test.txt"), 
        opts = opts_tune$min)
test_data <- test_set %>% 
  dplyr::select(userId, movieId) %>% 
  write.table(file = "test_data.txt", sep = " ", row.names = FALSE, col.names = FALSE)
test_data <- data_file(file.path(getwd(), "test_data.txt"), index1 = TRUE)
out_pred <- out_file(file.path(getwd(), "predict_test.txt"))
r$predict(test_data, out_pred)
pred_test <- read.csv2(file.path(getwd(), "predict_test.txt"), header = FALSE)
rating_predicted <- as.numeric(levels(pred_test[,1]))[pred_test[,1]] 
rmse_recosystem <- RMSE(test_set$rating, rating_predicted)
save(rmse_recosystem, file = "rmse_recosystem.Rda")
print(paste0("RMSE for latent factor model with recosystem package is ", rmse_recosystem))

