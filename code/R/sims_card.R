library(designmatch)
library(rdist)
library(sandwich)
library(stats)
library(tidyverse)

# set working directory to location of this file. If this does not work,
# set the working directory manually
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else {
  setwd(utils::getSrcDirectory()[1])
}

# function for computing norm
norm_vec <- function(x) sqrt(sum(x^2))

# function for computing features
features <- function(X, type="linear") {
  if (type == "linear"){
    return(X)
  } else if (type == "poly") {
    # polynomial feature formula
    formula <- as.formula(paste(' ~ .^2 -1 + ',
                                paste('poly(',colnames(X),',2, raw=TRUE)[, 2]',
                                      collapse = ' + ')
                                )
                          )
    # polynomial feature matrix
    X_poly <- model.matrix(formula, data=X)
    return(X_poly)
  }
}

# data variables
scenarios <- c("cw", "G")
types <- c("linear","poly")
names <- list(NULL,c("w1","w2","w3","w4","w5","w6","w7","w8",
                     "w9","w10","z.a","y.a"))
ntrials <- 1000
results <- data.frame(trial=integer(), 
                      scenario=character(), 
                      type=character(),
                      ate=double(), se=double(),
                      balance=double(), ss=integer(),
                      stringsAsFactors=FALSE)

# polynomial feature threshold vector
v <- rep(0.1, 65) 
v[1:10] <- 0.05

# loop through scenarios
for (scenario in scenarios) {
  # number of samples
  nsamples <- 500
  # true effect
  tau <- ifelse(scenario == "cw", 10, -0.4)
  # container for data
  simdata <- replicate(ntrials, 
                       data.frame(matrix(NA, nsamples, 12, dimnames=names)))
  
  # read in data
  for (t in 1:ntrials) {
    # filepath
    fpath = paste("../../data/simulations/", paste("data", scenario, t, sep="_"), ".csv", sep="")
    # get data frame and convert to list
    if (scenario == "cw"){
      simdata[, t] <- as.list(read.csv(fpath) %>% mutate_all(as.numeric) %>% 
                                rename(z.a = Z, y.a = Y, w1 = x1, w2 = x2, 
                                       w3 = x3, w4 = x4, w5 = x5, w6 = x6,
                                       w7 = x7, w8 = x8, w9 = x9, w10 = x10))
    } else {
      simdata[, t] <- as.list(read.csv(fpath) %>% mutate_all(as.numeric) %>% 
                                rename(z.a = Z, y.a = Y))
    }
  }
  
  # loop through balance conditions
  for (type in types){
    # containers
    ate <- rep(NA, ntrials)
    se <- rep(NA, ntrials)
    bal <- rep(NA, ntrials)
    ss <- rep(NA, ntrials)
    
    # threshold for balance
    scale <- ifelse(type == "linear", 0.01, 0.1)
    
    # loop through trials
    for (i in 1:ntrials) {
      # display trial every 50 trials
      if ((i-1) %% 50 == 0) {
        print(sprintf("%s (%s) : Trial %d of %d", scenario, type, i, ntrials))
      }
      
      cur.data <- as.data.frame(simdata[, i])
      # split into covariates and treatment
      X <- features(cur.data %>% dplyr::select(-"z.a",-"y.a"), type=type) 
      y <- cur.data$z.a # treatment
      out <- cur.data$y.a # outcome
      
      # sort X and y in descending order (of y)
      sort_ind <- order(y, decreasing = TRUE)
      X_sort <- X[sort_ind,]
      y_sort <- y[sort_ind]
      out_sort <- out[sort_ind] 
      
      # pooled sd
      pooled = sqrt((apply(X[y == 1,], 2 ,var) + apply(X[y == 0,], 2 ,var)) / 2)
      
      # build moments list for designmatch
      tols <- scale * pooled
      mom = list(covs = X_sort, tols = tols)
      
      # set up solver
      solver = list(name ="gurobi", t_max = 60, approximate = 1, trace_gurobi = 0)
      
      # solve
      match <- cardmatch(y_sort, mom = mom, fine = NULL, solver = solver)
      
      # check if nonzero solution found
      if (match$obj_total == 0) {
        print("no matches found")
        next
      }
      
      # weight vector
      weights <- rep(0.0, nsamples)
      weights[sort(c(match$t_id, match$c_id))] <- 1 / match$obj_total
      
      # regression (WLS equivalent to weighted DIM)
      # sandwich has issues with weights which are zero
      fit <- lm(out_sort[weights > 0] ~ y_sort[weights > 0], 
                weights=weights[weights > 0])
      
      # compute ate, se, and balance
      ate[i] <- fit$coefficients[2]
      se[i] <- sqrt(var(out_sort[y_sort == 1]) * sum(weights[y_sort == 1] ** 2) + 
                    var(out_sort[y_sort == 0]) * sum(weights[y_sort == 0] ** 2))
      bal[i] <- norm_vec(apply(X_sort[match$t_id, ], 2, mean) - 
                apply(X_sort[match$c_id, ], 2, mean))
      ss[i] <- 2*match$obj_total
    }
    
    # update results
    results <- rbind(results,
                     data.frame(trial=1:ntrials, 
                                scenario=rep(scenario, ntrials),
                                type=rep(type, ntrials),
                                ate=ate, se=se,
                                balance=bal, ss=ss,
                                stringsAsFactors=FALSE)
    )
  }
}

# save
write.csv(results,'../../results/sims_card.csv', row.names=FALSE)