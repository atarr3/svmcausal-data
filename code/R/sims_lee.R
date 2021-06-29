#~~~~~~~~~~~~~~~~~
 library(sandwich)
 library(stats)
 library(tidyverse)
 library(twang)
 library(rpart)
 library(ipred)
 library(randomForest)
 library(survey)
#~~~~~~~~~~~~~~~~~
 
 # global variables
 letters <- c("cw","G")
 methods <- c("LGR","RFRST")
 ate.wts <- c("lrg.ate.wts","rfrst.ate.wts")
 
 # set working directory to location of this file. If this does not work,
 # set the working directory manually
 if (interactive()) {
   setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
 } else {
   setwd(utils::getSrcDirectory()[1])
 }

# function computes the ASAM for the gbm model after "i" iterations 
# x is a data frame with only the covariates 
# z is a vector of 0s and 1s indicating treatment assignment 
# weight.type is "ATT" or "ATE"
    F.asam.iter <- function(i,ps.model,x,z,weight.type) { 
        i <- floor(i) # makes sure that i is an integer  
        
        # predict(ps.model, x, i) provides predicted values on the log-odds of 
        #  treatment for the gbm model with i iterations at the values of x 
        odds <- exp(predict(ps.model, x, i))
        pscores <- odds/(1+odds)
        
        # create temp weights
        w <- rep(1, nrow(x))
            
        if (weight.type=="ATE") {
                w <- ifelse(z==1, 1/pscores, 1/(1-pscores))
        } else
        
        {
                w <- ifelse(z==1, 1, pscores/(1-pscores))
        } 
        
        # sapply repeats calculation of F.std.diff for each variable (column) of x 
        # this calculates ASAM -- the mean of the  
        # standardized differences for all variables in x 
        asam <- mean(unlist(sapply(x, F.std.diff, z=z, w=w)))
        #cat(i,asam,"\n") 
        return(asam) 
    }  
 
# function: logistic regression estimation of propensity score

    F.lrg.sim <- function(dataset) {
        ps.model <- glm(z.a ~ w1+w2+w3+w4+w5+w6+w7+w8+w9+w10, data=dataset, family=binomial(link="logit"))
        odds <- exp(predict(ps.model))
            
        # saves propensity scores
        PSCORES <<- odds/(1+odds)
    }

# function: random forests estimation of propensity scores
    F.rfrst.sim <- function(dataset) {
        ps1 <- 1
        while (max(ps1)> .999) {
            gc(verbose=TRUE)
            ps.model <- randomForest(z.a ~ w1+w2+w3+w4+w5+w6+w7+w8+w9+w10, data=dataset)
            ps1 <- ps.model$votes[, 2]
        }

        # saves propensity scores
        PSCORES <<- ps1        
    }

# function: calculate ASAM for weighted data
# ASAM: the average standardized absolute mean difference in the covariates

    F.std.diff <- function(u,z,w) 
        { 
        # for variables other than unordered categorical variables compute mean differences 
        # mean(u[z==1]) gives the mean of u for the treatment group 
        # weighted.mean() is a function to calculate weighted mean 
        # u[z==0],w[z==0] select values of u and the weights for the comparison group 
        # weighted.mean(u[z==0],w[z==0],na.rm=TRUE): weighted mean for the comparison group 
        # sd(u[z==1], na.rm=T) calculates the standard deviation for the treatment group 
            
        if(!is.factor(u)) 
        { 
            sd1 <- sd(u[z==1], na.rm=T) 
            if(sd1 > 0) 
            { 
                result <- abs(mean(u[z==1],na.rm=TRUE)- 
                                weighted.mean(u[z==0],w[z==0],na.rm=TRUE))/sd1 
            } else 
            { 
                result <- 0 
                warning("Covariate with standard deviation 0.") 
            } 
        } 
        
        # for factors compute differences in percentages in each category 
        # for(u.level in levels(u) creates a loop that repeats for each level of  
        #  the categorical variable 
        # as.numeric(u==u.level) creates as 0-1 variable indicating u is equal to 
        #  u.level the current level of the for loop 
        # std.diff(as.numeric(u==u.level),z,w)) calculates the absolute  
        #   standardized difference of the indicator variable 
        else 
        { 
            result <- NULL 
            for(u.level in levels(u)) 
            { 
                result <- c(result, std.diff(as.numeric(u==u.level),z,w)) 
            } 
        } 
        return(result) 
    }
    
    # function: calculate effect betas, SEs, bias, CI coverage #also MSE?
    F.calculate <- function(g1, sim) {
      # useful quantities
      outcomes <- sim$y.a
      treat <- sim$z.a
      n <- length(treat)
      weights <- sim$weights / n
      
      # g1 is true ATE
      temp.design <- svydesign(ids=~1, weights=~weights, data=sim)
      temp.glm <- svyglm(y.a~z.a, design=temp.design)
      beta <- summary(temp.glm)$coef[2]
      bias <- (g1-beta)/g1 # relative
      abs.bias <- abs(bias)
      fit <- lm('y.a ~ z.a', weights=sim$weights, data=sim)
      std.err <- summary(temp.glm)$coef[4]
      class(temp.glm) <- "glm"
      temp.cover <- confint(temp.glm)
      ci.cover <- ifelse(temp.cover[2, 1] < g1 & temp.cover[2, 2] > g1, 1, 0)   
      asam <- mean(unlist(sapply(sim[,-c(11:13)], F.std.diff, z=sim$z.a, w=sim$weights))) 
      sq.err <- (g1-beta)^2
      return(c(beta, std.err, ci.cover, bias, abs.bias, asam, sq.err, std.err2))
    }

#~~~~~~~~~~~~~~~~~
# CALLS
# and weight construction
    
# data variables
scenarios <- c("cw", "G")
names <- list(NULL,c("w1","w2","w3","w4","w5","w6","w7","w8",
                     "w9","w10","z.a","y.a"))
ntrials <- 1000
results <- data.frame(trial=integer(), 
                      scenario=character(), 
                      method=character(),
                      ate=double(), se=double(), 
                      stringsAsFactors=FALSE)

# loop through sims
for (scenario in scenarios) {
  # number of samples
  nsamples <- ifelse(scenario == "cw", 200, 500)
  # true effect
  tau <- ifelse(scenario == "cw", 10, -0.4)
  # container for data
  simdata <- replicate(ntrials, 
                       data.frame(matrix(NA, nsamples, 12, dimnames=names)))
  
  # loop through trials
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
  
  # logistic regression
  lrg.pscores <- lrg.att.wts <- lrg.ate.wts <- NULL
  for (i in 1:dim(simdata)[2]) {
    sim <- as.data.frame(simdata[, i])
    cat(i,": ", "\n")
    F.lrg.sim(sim)
    ATT.WTS <- ifelse(sim$z.a==1, 1, PSCORES/(1-PSCORES))
    ATE.WTS <- ifelse(sim$z.a==1, 1/PSCORES, 1/(1-PSCORES))
    lrg.pscores <- cbind(lrg.pscores, PSCORES)
    lrg.att.wts <- cbind(lrg.att.wts, ATT.WTS)
    lrg.ate.wts <- cbind(lrg.ate.wts, ATE.WTS)
  }
  
  # random forests
  rfrst.pscores <- rfrst.att.wts <- rfrst.ate.wts <- NULL
  for (i in 1:dim(simdata)[2]) {
    sim <- as.data.frame(simdata[, i])
    sim$z.a <- as.factor(sim$z.a)
    cat(i,": ", "\n")
    F.rfrst.sim(sim)
    ATT.WTS <- ifelse(sim$z.a==1, 1, PSCORES/(1-PSCORES))
    ATE.WTS <- ifelse(sim$z.a==1, 1/PSCORES, 1/(1-PSCORES))
    rfrst.pscores <- cbind(rfrst.pscores, PSCORES)
    rfrst.att.wts <- cbind(rfrst.att.wts, ATT.WTS)
    rfrst.ate.wts <- cbind(rfrst.ate.wts, ATE.WTS)
  }
  
  # get results
  WEIGHTS <- ate.wts #specify ATE or ATT weights
  temp.b <- temp.se <- temp.se2 <- temp.ci <- temp.bias <- temp.abs.bias <- temp.asam <- temp.sq.err <- matrix(data=NA, ncol=length(WEIGHTS), nrow=dim(simdata)[2])
  colnames(temp.b) <- colnames(temp.se) <- colnames(temp.ci) <- colnames(temp.bias) <- colnames(temp.abs.bias) <- colnames(temp.asam) <- colnames(temp.sq.err) <- methods
  for (j in 1:length(WEIGHTS)) { # for each of the PS methods LRG thru RFRST
    for (i in 1:dim(simdata)[2]) { # for each of 1000 scenario datasets
      temp.WTS <- eval(as.name(WEIGHTS[j]))
      sim <- as.data.frame(simdata[, i])
      sim$weights <- temp.WTS[, i]
      temp.results <- F.calculate(tau, sim)
      temp.b[i, j]  <- temp.results[1]
      temp.se[i, j] <- temp.results[2]
      temp.ci[i, j] <- temp.results[3] 
      temp.bias[i, j] <- temp.results[4]
      temp.abs.bias[i, j] <- temp.results[5] 
      temp.asam[i, j] <- temp.results[6]
      temp.sq.err[i, j] <- temp.results[7]
      cat(WEIGHTS[j],i,": ", temp.results, "\n")
    }
  }
  
  # store results
  results <- rbind(results,
                   data.frame(trial=rep(1:ntrials, 2), 
                              scenario=rep(scenario, length(methods)*ntrials),
                              method=rep(methods, each=ntrials),
                              ate=c(temp.b), se=c(temp.se),
                              stringsAsFactors=FALSE)
                   )
}

# save
write.csv(results,'../../results/sims_lee.csv', row.names=FALSE)
