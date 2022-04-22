##########################################################################
# Building a simple neural network

# import spreadsheet files
library(readr)
# deep learning package
library(keras)
# dynamic interactive tables
library(DT)

data <- read_csv("SimulatedBinaryClassificationDataset.csv",
                 col_names = TRUE)
summary(data)

# data.frame --> matrix
data  <- as.matrix(data)
# remove the row and col names, leaving only numerical values
dimnames(data) = NULL
mode(data)

# train and test split index
set.seed(123)
index <- sample(2,
                nrow(data),
                replace = TRUE,
                prob = c(0.9, 0.1))
table(index)

# data splitting
x_train <- data[index == 1, 1:10]
x_test <- data[index == 2, 1:10]
y_test_actual <- data[index == 2, 11]

# use teh to_categorical function in keras package for one-hot encoding
y_train <- to_categorical(data[index == 1, 11])
y_test <- to_categorical(data[index == 2, 11])

model <- keras_model_sequential()

model %>%
  # layer_dense means a densely connected layer
  layer_dense(name = "DeepLayer1",
              units = 10, # hyperparameter: the number of nodes
              activation = "relu",
              # the first layer need to have specification about the input dimension
              input_shape = c(10)) %>%
  layer_dense(name = "DeepLayer2",
              units = 10,
              activation = "relu") %>%
  layer_dense(name = "OutputLayer",
              units = 2,
              # softmax function will provide probabilities of the nodes
              activation = "softmax")

summary(model)

model %>% compile(
  # another way to calculate loss, besides mean-squared-error
  loss = "categorical_crossentropy", 
  # a special way of gradient descent
  optimizer = "adam",
  # measurement of model performance - using accuracy to measure
  metrics = c("accuracy")) 

history <- model %>%
  fit(x_train,
      y_train,
      # number of full forward & backward propagation
      # (i.e. run 10 times back and forth of all samples)
      epoch = 10, 
      # instead of propagating the whole dataset at one go, use smaller batches
      batch_size = 256, 
      # splitting the training set to test itself during training
      validation_split = 0.1, 
      verbose = 2)

# plot the training history
plot(history)

model %>%
  evaluate(x_test,
           # NOTE: here we are still using the one-hot encoded y_test
           y_test)

# form predictions
pred <- model %>%
  predict(x_test) %>%
  k_argmax()
# reference for converting tensor to R data types
# https://torch.mlverse.org/technical/tensors/
pred <- as.array(pred)
table(Predicted = pred,
      # NOTE: for confusion matrix, we are using the original y_test_actual, not encoded
      Actual = y_test_actual)
##########################################################################
# Applying regularization to deal with overfitting
library(keras)
library(readr)
library(tidyr)
library(tibble)
library(plotly)

# specify the number of feature variables for the dataset to be downloaded
num_words <- 5000
imdb <- dataset_imdb(num_words = num_words)

# train test split
c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test
# training and test data are both stored as lists
length(train_data)

# multi-hot encoding
multi_hot_sequences <- function(sequences, dimension){
  multi_hot <- matrix(0, 
                      # the number of samples in the sequences
                      # sequences are stored as lists
                      nrow = length(sequences), 
                      ncol = dimension)
  for(i in 1 : length(sequences)){
    # sequences[[i]] extracts the label of the words in the text sample i
    # which ever word is included in that sequence will be assigned 1 at row i
    multi_hot[i, sequences[[i]]] <- 1
  }
  multi_hot
}

train_data <- multi_hot_sequences(train_data, num_words)
test_data <- multi_hot_sequences(test_data, num_words)

baseline_model <-
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

baseline_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

baseline_model %>% summary()

baseline_history <- baseline_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  # recall: batch size controls the number of training samples to work through before updating parameters
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  # get loss and accuracy reports
  verbose = 2
)

plot(baseline_history)

l2_model <-
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words,
              # apply regularization in the layer_dense function's argument
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 16, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 1, activation = "sigmoid")

l2_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

l2_model %>% summary()

l2_history <- l2_model %>% fit(
  train_data,
  train_labels,
  epoch = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

plot(l2_history)

drop_model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  # a new layer to specify the dropout rate
  layer_dropout(0.6) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 1, activation = "sigmoid")

drop_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

drop_model %>% summary()

drop_history <- drop_model %>% fit(
  train_data,
  train_labels,
  epoch = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

plot(drop_history)
##########################################################################
# Optimization of neural nets
library(keras)
library(readr)

train.import <- read_csv("ImprovementsTrain.csv")
test.import <- read_csv("ImprovementsTest.csv")

# recall that NN are constructed based on numerical matrices
# we need to cast dataframe into matrix and remove column names
train.import <- as.matrix(train.import)
dimnames(train.import) <- NULL
test.import <- as.matrix(test.import)
dimnames(test.import) <- NULL

# train & test sets
train_data <- train.import[, 1:12]
train_labels <- train.import[, 13]
test_data <- test.import[, 1:12]
test_labels <- test.import[, 13]

feature.means = vector(length = ncol(train_data))
for(i in 1:length(feature.means)){
  # calculate the mean of each column in the training set
  feature.means[i] = mean(train_data[, i])
}

feature.sds = vector(length = ncol(train_data))
for(i in 1:length(feature.sds)){
  # calculate the standard deviation of each column in the training set
  feature.sds[i] <- sd(train_data[, i])
}

# normalize the feature variables in the training set
train_data_normalized <- matrix(nrow = nrow(train_data),
                                ncol = ncol(train_data))
for(n in 1:ncol(train_data)){
  for(m in 1:nrow(train_data)){
    train_data_normalized[m, n] <- (train_data[m, n] - feature.means[n])/feature.sds[n]
  }
}

# normalize the feature variables in the testing set
test_data_normalized <- matrix(nrow = nrow(test_data),
                               ncol = ncol(test_data))
for(n in 1:ncol(test_data)){
  for(m in 1:nrow(test_data)){
    test_data_normalized[m, n] <- (test_data[m, n] - feature.means[n])/feature.sds[n]
  }
}

# use normal distribution to set the very first weights in tensor
init_w <- initializer_random_normal(mean = 0,
                                    stddev = 0.05,
                                    seed = 123)
# by default, all the bias terms are set to 0 initially
init_B <- initializer_zeros()

baseline_model <- keras_model_sequential() %>%
  layer_dense(units = 48,
              activation = "relu",
              # the initial weights for the NN
              kernel_initializer = init_w,
              input_shape = c(12)) %>%
  layer_dense(units = 48,
              activation = "relu") %>%
  layer_dense(units = 1,
              activation = "sigmoid")

summary(baseline_model)

baseline_model %>% compile(
  optimizer = optimizer_rmsprop(
    # learning rate (i.e. step size)
    lr = 0.001,
    # the decay factor (i.e. the weight to the previous gradient, beta)
    rho = 0.9
  ),
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

baseline_history <- baseline_model %>%
  fit(train_data_normalized,
      train_labels,
      epochs = 40,
      # conditions when to stop, save computation time
      # avoid scenario of training without improvements
      callbacks = list(callback_early_stopping(
        # use change in loss to determine whether to stop
        monitor = "loss",
        # wait for two runs, if unchanged in loss, then stop
        patience = 2
      )),
      batch_size = 512,
      validation_data = list(test_data_normalized, test_labels),
      verbose = 2)

plot(baseline_history)
##########################################################################
















