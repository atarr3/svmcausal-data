library(designmatch)
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

# function for computing norm of vector
norm_vec <- function(x) sqrt(sum(x^2))

# feature types
types <- c("linear", "poly")

# read in conditional variances
cv1 <- read.csv("../../results/cv1.csv", header=FALSE)$V1
cv0 <- read.csv("../../results/cv0.csv", header=FALSE)$V1

for (type in types) {
  # read in data
  if (type == "linear") {
    rhc <- read.csv("../../data/empirical/rhc_clean.csv")
  } else {
    rhc <- read.csv("../../data/empirical/rhc_poly_clean.csv") 
  }
  
  # data features
  nsamples <- nrow(rhc)
  
  # split into covariates and treatment
  X <- rhc %>% dplyr::select(-"death",-"swang1")
  y <- rhc$swang1 # treatment
  out <- rhc$death # outcome
  
  # number of treated and control
  n1 <- sum(y)
  n0 <- length(y) - n1
  
  # sort X and y in descending order (of y)
  X_sort <- X[order(y, decreasing = TRUE),]
  y_sort <- y[order(y, decreasing = TRUE)]
  out_sort <- out[order(y, decreasing = TRUE)] 
  
  # pooled sd
  pooled <- sqrt((apply(X[y == 1,], 2 ,var) + apply(X[y == 0,], 2 ,var)) / 2)
  
  # build moments list for designmatch
  tols <- 0.1 * pooled
  mom <- list(covs = X_sort, tols = tols)
  
  # set up solver
  solver <- list(name ="gurobi", t_max = 300, approximate = 0, trace_gurobi = 1)
  
  # solve
  match <- cardmatch(y_sort, mom = mom, fine = NULL, solver = solver)
  
  # corresponding weight vector
  weights <- rep(0, nsamples)
  weights[sort(c(match$t_id, match$c_id))] <- 1 / match$obj_total
  
  # regression estimate
  fit <- lm(out_sort[weights > 0] ~ y_sort[weights > 0], 
            weights=weights[weights > 0])
  
  # ate and se
  ate <- fit$coefficients[2]
  se <- sqrt(sum(weights[y_sort == 1] ** 2 * cv1) + 
             sum(weights[y_sort == 0] ** 2 * cv0))
  
  # balance measures
  sdim <- (apply(X_sort[match$t_id,], 2, mean) - 
           apply(X_sort[match$c_id,], 2, mean)) / pooled
  X_sort <- X_sort %>% mutate_all(scale)
  # normed difference in means
  bal <- norm_vec(apply(X_sort[match$t_id,], 2, mean) - 
                  apply(X_sort[match$c_id,], 2, mean))
  
  # subset size
  ss <- 2*match$obj_total
  
  # effective subset size
  ess <- ss
  
  # store results
  results <- data.frame(ate=ate,
                        se=se,
                        bal=bal,
                        bal_sd=mean(abs(sdim)),
                        ss=ss,
                        ess=ess,
                        row.names=NULL)
  
  # fix variable names
  varnames <- names(sdim)
  
  # for (i in 1:length(varnames)) {
  #   if (grepl("\\.", varnames[i]) & !grepl("wtkilo1", varnames[i])) {
  #     # replace all the garbage with something readable
  #     varnames[i] <- gsub("\\."," ", varnames[i])
  #   }
  # }
  
  results[varnames] <- sdim
  
  # save
  write.csv(results, sprintf('../../results/rhc_card_%s.csv', type), 
            row.names=FALSE)
}
