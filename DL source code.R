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