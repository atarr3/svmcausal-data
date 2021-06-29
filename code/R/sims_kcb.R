library("ATE.ncb")
library(rdist)
library(sandwich)
library(stats)
library(survey)
library(tidyverse)

# set working directory to location of this file. If this does not work,
# set the working directory manually
if (interactive()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else {
  setwd(utils::getSrcDirectory()[1])
}

kcb <- function(X, y, outcomes, kernel="linear") {
  # number of samples
  n <- length(y)
  # treatment indicator
  treat <- as.numeric(y == 1)
  
  # compute gram matrix
  if (kernel == "linear"){
    K = X %*% t(X)
  } else if (kernel == "rbf") {
    # pairwise distances
    pd <- pdist(X, metric="euclidean")
    # compute length scale parameter via median euclidean distance
    gamma <- median(pd)
    # kernel matrix
    K <- exp(- pd^2 / gamma^2)
  } else if (kernel == "poly") {
    # polynomial feature formula
    formula <- as.formula(paste(' ~ .^2 -1 + ',
                                paste('poly(',colnames(X),',2, raw=TRUE)[, 2]',
                                      collapse = ' + ')
                                )
                          )
    # polynomial feature matrix
    X_poly <- model.matrix(formula, data=data.frame(X))
    K = X_poly %*% t(X_poly)
  }
  
  # design a grid for the tuning parameter
  nlam <- 50
  lams <- exp(seq(log(1e-8), log(1), len=nlam))
  
  # compute weights for T=1
  fit1 <- ATE.ncb.SN(treat, K, lam1s=lams, traceit=FALSE)
  
  # compute weights for T=0
  fit0 <- ATE.ncb.SN(1-treat, K, lam1s=lams, traceit=FALSE)
  
  # unable to compute weights, return NA
  if (any(is.na(fit1$w)) | any(is.na(fit0$w))) {
    return(NA)
  }
  
  # lambda at bound
  # if (sum(fit0$warns) | sum(fit1$warns)) cat("lambda bound warning!\n")
  
  # DIM estimate
  ate <- mean(fit1$w*outcomes - fit0$w*outcomes)
  
  # weight vector
  weights <- fit1$w + fit0$w
  
  # survey estimates for se
  design <- svydesign(ids=~1, weights=weights, data=data.frame(y.a=outcomes,
                                                               z.a=treat))
  glm <- svyglm(y.a~z.a, design=design)
  se2 <- summary(glm)$coef[4]
  
  # linear model and standard error
  weights <- weights / n
  se <- sqrt(var(outcomes[treat == 1]) * sum(weights[treat == 1] ** 2) +
             var(outcomes[treat == 0]) * sum(weights[treat == 0] ** 2))
  
  return(list(ate=ate, se=se, se2=se2))
}

# compute ate for simulations
ntrials <- 1000
# cw = Chan & Wong, A, E, G = Setoguchi scenario A, E, G
scenarios <- c("cw","G") 
kernels <- c("rbf")

# initialize results data frame
results <- data.frame(trial=integer(),
                      scenario=character(),
                      ate=double(),
                      se=double(),
                      se2=double(),
                      stringsAsFactors=FALSE)

# iterate through simulations
for(scenario in scenarios) {
  # set true effect
  tau <- ifelse(scenario == "cw", 10, -0.4)
  
  # iterate throught kernels
  for (kernel in kernels) {
    # initialize ate vector
    ate <- rep(NA, ntrials)
    se <- rep(NA, ntrials)
    se2 <- rep(NA, ntrials)
    # iterate through trials
    for (t in 1:ntrials) {
      # display trial every 50 trials
      if ((t-1) %% 50 == 0) {
        print(sprintf("%s (%s) : Trial %d of %d", scenario, kernel, t, ntrials))
      }
      
      # read in data
      dpath <- paste(paste('../../data/simulations/data', scenario, t, sep='_'), '.csv', sep='')
      data <- read.csv(dpath)
      # split into X, y, and outcomes
      X <- as.matrix(data %>% select(-"Z", -"Y") %>% mutate_all(scale))
      y <- data$Z
      outcomes <- data$Y
      
      # compute balancing weights and estimate ate
      res <- kcb(X, y, outcomes, kernel)
      ate[t] <- res$ate
      se[t] <- res$se
      se2[t] <- res$se2
    }
    
    # failures
    print(sprintf("Scenario '%s' with kernel '%s' failed on %d trials", 
                  scenario, kernel, sum(is.na(ate)) 
                  )
          )
    
    # performance
    bias <- ate - tau
    rmse <- sqrt(mean(bias^2, na.rm=TRUE))
    
    # update data frame
    results <- rbind(results, data.frame(trial=1:ntrials,
                                         scenario=rep(scenario,ntrials),
                                         ate=ate,
                                         se=se,
                                         se2=se2,
                                         stringsAsFactors=FALSE))
    
    # print out results
    print(sprintf("Results for Scenario %s (%s)", scenario, kernel))
    print(sprintf(" ATE: %.2f", mean(ate, na.rm=TRUE) ))
    print(sprintf("Bias: %.2f", mean(bias, na.rm=TRUE) ))
    print(sprintf("RMSE: %.2f", rmse))
  }
}

# save results
write.csv(results,'../../results/sims_kcb.csv', row.names=FALSE)