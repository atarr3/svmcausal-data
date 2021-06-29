library("ATE.ncb")
library(rdist)
library(iWeigReg)
library(stats)
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
  
  # number of treated and control
  n1 <- sum(treat)
  n0 <- n - n1
  
  # read in conditional variances
  cv1 <- read.csv("../../results/cv1.csv", header=FALSE)$V1
  cv0 <- read.csv("../../results/cv0.csv", header=FALSE)$V1
  
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
  fit1 <- ATE.ncb.SN(treat, K, lam1s=lams, traceit=TRUE)
  
  # compute weights for T=0
  fit0 <- ATE.ncb.SN(1-treat, K, lam1s=lams, traceit=TRUE)
  
  # unable to compute weights, return NA
  if (any(is.na(fit1$w)) | any(is.na(fit0$w))) {
    return(NA)
  }
  
  # lambda at bound
  # if (sum(fit0$warns) | sum(fit1$warns)) cat("lambda bound warning!\n")
  
  # DIM estimate
  ate1 <- mean(fit1$w*outcomes - fit0$w*outcomes)
  
  # weight vector
  weights <- (fit1$w + fit0$w) / n
  
  # normalized weights (sum to n1 or n0)
  w1 <- fit1$w[treat == 1] / sum(fit1$w[treat == 1]) * n1
  w0 <- fit0$w[treat == 0] / sum(fit0$w[treat == 0]) * n0
  
  # outcomes
  out1 <- outcomes[treat == 1]
  out0 <- outcomes[treat == 0]
  
  # Horvitz-Thompson estimate
  ate2 <- sum(w1 * out1) / n1 - sum(w0 * out0) / n0
  
  # weighted Neyman estimate
  se1 <- sqrt(sum(w1 ** 2 * cv1) / n1 ** 2 + sum(w0 ** 2 * cv0) / n0 ** 2)
  
  # standard error (Horvitz-Thompson estimate)
  props <- 1 / (weights * n)
  ht <- ate.HT(outcomes, treat, props) 
  
  # se <- sqrt(sum(weights[treat == 1] ** 2) +
  #            sum(weights[treat == 0] ** 2))
  se2 <- sqrt(ht$v.diff)

  # balance (NDIM)
  w <- weights
  tp = 2*treat - 1
  bal <- sqrt(w %*% (K * (tp %o% tp)) %*% w)
  
  # effective subset size
  ess <- sum(fit1$w)^2 / sum(fit1$w^2) + sum(fit0$w)^2 / sum(fit0$w^2)
  
  return(list(ate=ate1, ate_ht=ate2, se=se1, se_ht=se2, bal=bal, ess=ess, weights=w))
}

# read in data
rhc <- read.csv("../../data/empirical/rhc_clean.csv")

# split into covariates and treatment
X <- rhc %>% dplyr::select(-"death",-"swang1") %>% mutate_all(scale)
y <- rhc$swang1 # treatment
outcomes <- rhc$death # outcome

res <- kcb(X, y, outcomes, "rbf")

results <- data.frame(ate=res$ate,
                      ate_ht=res$ate_ht,
                      se=res$se,
                      se_ht=res$se_ht,
                      bal=res$bal,
                      ess=res$ess,
                      stringsAsFactors=FALSE)

# save results
write.csv(results,'../../results/rhc_kcb_rbf.csv', row.names=FALSE)