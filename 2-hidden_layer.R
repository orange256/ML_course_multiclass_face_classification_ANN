# -- Machine Learning 2017 -- 
# [HW 3] : Neural Network (for classification)
# [Deadline] : 2017/4/25
# [Model] : 2 hidden layer with stochastic gradient descent
# [Activation function] : rectified function (f(x) = max{0,x})

#==================================================================================
# Functions 
#==================================================================================

# [get_Training_Matrix] ----------------------------------------------------------
  # This function can get picture's pixels, and save as dataframe.
get_Training_Matrix<- function(class,num){
  M <- matrix(0,1,900) # 30x30 pixels
  for(i in 1:num){
    tmp <- read.bmp(paste0(path,"/Data_Train/Class",class,"/faceTrain",class,"_",i,".bmp")) %>% 
      t %>% as.vector() 
    M <- rbind(M,tmp)
  }
  M <- as.data.frame(M)
  M <- data.frame(class = rep(paste0("C",class),num),
                  num = 1:num) %>% cbind.data.frame(.,M[-1,])
  return(M)
}

# [my_2_hidden_model] ----------------------------------------------------------
  # This function is for Error Back-propagation.
  # you need to choose "initial weight" , and learning rate "eta"
my_2_hidden_model <- function(phi, M1, M2, eta, weight_1, weight_2, weight_3){
  
  for(i in 1:nrow(phi)){
    # 1st hidden layer
    # a_j & z_j
    a_j <- weight_1 %*% phi[i,] 
    z_j <- rbind(1 , as.matrix(sapply(a_j,function (x) max(0,x))) ) # add z_0 = 1
    
    # a_l
    a_l <- weight_2 %*% z_j
    
    # 2nd hidden layer
    # z_l
    z_l <- rbind(1 , as.matrix(sapply(a_l,function (x) max(0,x))) ) 
    
    # a_k
    a_k <- weight_3 %*% z_l
    
    # y_k (Soft Max)
    y_k <- exp(a_k) / sum(exp(a_k))
    
    # delta_k
    delta_k <- y_k - T_matrix[i,]
    
    
    
    # delta_l
    # h'(a_k) = max{0,1}
    delta_l <- (as.matrix(sapply(a_l,function (x) max(0,x))) / a_l) * (t(weight_3[,c(2:(M2+1))] ) %*% delta_k )
    
    
    # delta_j
    # h'(x) = max{0,1}
    delta_j <- (as.matrix(sapply(a_j,function (x) max(0,x))) / a_j) * (t(weight_2[,c(2:(M1+1))] ) %*% delta_l )
    
    # partial En / partial W3
    gradient_W3 <- delta_k %*% t(z_l)
    
    # partial En / partial W2
    gradient_W2 <- delta_l %*% t(z_j)
    
    # partial En / partial W1
    gradient_W1 <- delta_j %*% phi[i,]
    
    # Stochastic Gradient Desent
    weight_1 <- weight_1 - eta*gradient_W1
    weight_2 <- weight_2 - eta*gradient_W2
    weight_3 <- weight_3 - eta*gradient_W3
    
  }
  return(list(weight_1,weight_2,weight_3))
}

# [get_2_layer_output] ------------------------------------------------------
  # This function will help you to get output through 2-layer model
get_2_layer_output <- function(data, W1, W2, W3){
  output <- c(999,999,999) %>% as.matrix() %>% t
  for(i in 1:nrow(data)){
    
    # 1st hidden layer
    # a_j & z_j
    a_j <- W1 %*% data[i,] 
    z_j <- rbind(1 , as.matrix(sapply(a_j,function (x) max(0,x))) ) # add z_0 = 1
    
    # a_l
    a_l <- W2 %*% z_j
    
    # 2nd hidden layer
    # z_l
    z_l <- rbind(1 , as.matrix(sapply(a_l,function (x) max(0,x))) ) 
    
    # a_k
    a_k <- W3 %*% z_l
    
    # y_k (Soft Max)
    y_k <- exp(a_k) / sum(exp(a_k))
    
    output <- rbind(output,t(y_k))
  }
  
  return(output[-1,])
}


#====================================================================================
# Main 
#====================================================================================

# [0] path & packages ---------------------------------------------------------------
path <- c("D:/Google Drive/NCTU/106/下學期/機器學習/HW/HW3/") # Windows
path <- c("/Users/bee/Google Drive/NCTU/106/下學期/機器學習/HW/HW3/") # Mac

library(bmp)
library(magrittr)
library(dplyr)
library(ggplot2)
library(sigmoid)

# [1] get row data --------------------------------------------------------------------
train_raw <- rbind.data.frame(get_Training_Matrix(1,1000),
                              get_Training_Matrix(2,1000),
                              get_Training_Matrix(3,1000))

# seperating data ( train:valid = 9:1 ) -------------------------------------------
# we use "balance" data setting (mod 10)
D_train <- train_raw %>% filter((num %% 10) != 0)
D_valid <- train_raw %>% filter((num %% 10) == 0)

# [2] PCA ---------------------------------------------------------------------------
# get PCA parameters from training data
PCA <- prcomp(D_train[,c(3:902)], scale = TRUE)
D_train_PCA <- cbind.data.frame(D_train[,1:2],as.data.frame(PCA$x[,1:2]))

# get PCA result for validation data  
valid_PCA_data <- predict(PCA, D_valid[,c(3:902)])
D_valid_PCA <- cbind.data.frame(D_valid[,1:2],as.data.frame(valid_PCA_data[,1:2]))


# [3] Neural network model -------------------------------------------------
# Initial setting

# phi (2700x2)
phi <- D_train_PCA[,c(3,4)] %>% as.matrix()
# add constant node => (2700x3)
phi <- cbind( constant = c(rep(1, nrow(phi))) , phi)

# true value matrix (2700x3)
T_matrix <- rbind(matrix(rep(c(1,0,0),each=900),900,3),
                  matrix(rep(c(0,1,0),each=900),900,3),
                  matrix(rep(c(0,0,1),each=900),900,3))

# M : number of nodes
M1 <- 10
M2 <- 10
# initial weight W1 & W2
set.seed(9487)
weight_1 <- matrix(runif(3*M1  ,-1,1), M1, 3) # M1 x 3
weight_2 <- matrix(runif((M2)*(M1+1),-1,1), M2, M1+1) # (M2+1) x (M1+1)
weight_3 <- matrix(runif(3*M2+3,-1,1), 3, M2+1) # 3 x (M2+1)

# Error Back-Propagation (start) ----------------------------------------------------
# In order to get better weight, you can do this several times.(depends on eta)
# do 5 times, hit rate = .996
new_weight_vectors <- my_2_hidden_model(phi = phi,
                                        M1 = M1,
                                        M2 = M2,
                                        eta = 0.01,
                                        weight_1 = weight_1,
                                        weight_2 = weight_2,
                                        weight_3 = weight_3)

weight_1 <- new_weight_vectors[[1]]
weight_2 <- new_weight_vectors[[2]]
weight_3 <- new_weight_vectors[[3]]

# Error Back-Propagation (end) ---------------------------------------------------

# [4] validataion data ---------------------------------------------------------------
  phi_valid <- D_valid_PCA[,c(3,4)] %>% as.matrix()
  # add constant node => (2700x3)
  phi_valid <- cbind( constant = c(rep(1, nrow(phi_valid))) , phi_valid)

  validation_output <- get_2_layer_output(data = phi_valid,
                                          W1 = weight_1,
                                          W2 = weight_2,
                                          W3 = weight_3)
  
  validation_output <- as.data.frame(validation_output) %>% cbind.data.frame(D_valid_PCA,.)
  
  # choose biggest one 
  for(i in 1:nrow(validation_output)){
    validation_output$which_is_max[i] <- which.max(validation_output[i,5:7])
  }
  
  hit_table <- table(validation_output$class, validation_output$which_is_max)
  hit_table
  
  hit_rate <- sum(diag(hit_table)) / nrow(validation_output)
  hit_rate
  
  error_rate <- 1 - hit_rate
  error_rate
  
  
  
# [5] Plots ----------------------------------------------------------------------------
# training data
  ggplot(D_train_PCA,aes(x=PC1, y=PC2, group = class) ) + 
    geom_point(aes(color = class, shape = class))+
    xlim(-40,40)+
    ylim(-40,40)+
    stat_ellipse()
  
# Decision region
  # generate fake data 
  decision <- data.frame(PC1 = runif(10000,-40,40),
                         PC2 = runif(10000,-40,40)) 
  
  phi_decision <- decision %>% as.matrix()
  # add constant node => (2700x3)
  phi_decision <- cbind( constant = c(rep(1, nrow(phi_decision))) , phi_decision)
  
  decision_output <- get_2_layer_output(data = phi_decision,
                                        W1 = weight_1,
                                        W2 = weight_2,
                                        W3 = weight_3)
  
  decision_output <- as.data.frame(decision_output) %>% cbind.data.frame(decision,.)
  
  # choose biggest one 
  for(i in 1:nrow(decision_output)){
    decision_output$which_is_max[i] <- which.max(decision_output[i,3:5])
  }
  
  decision_output$which_is_max <-as.factor(decision_output$which_is_max)
  
  # plot decision region
  ggplot(decision_output,aes(x=PC1, y=PC2, group = which_is_max) ) + 
    geom_point(aes(color = which_is_max, shape = which_is_max))
  
  

  
  