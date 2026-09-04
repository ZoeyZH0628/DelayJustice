###########################################################################
# DJ_rstan_fit_v2.R                                                       #
# Author: Yang Hu                                                         #
# Date: Oct. 16, 2022                                                     #
# Date of last change: Feb. 22, 2022                                      #
# Project: Delay Justice (DJ)                                             #                           
# PI: Dr. Yang Hu (ECNU)                                                  #
# Collaborators:                                                          #
# Dr. Lisheng He (SHU)                                                    #
# Zexi Wang (ECNU)                                                        #
#                                                                         #
# This is a custom R function to estimate the parameters of subjective    #
# value functions with Hierarchical Bayesian Modelling (hbm).             #
###########################################################################

library(readxl)
library(tidyverse) 
library(rstan)
library(ggplot2)
library(loo)

# library(dineq) 

# main function #
DJ_rstan_fit <- function(modelStr, test = NULL, saveFit = NULL, suffix = NULL, 
                         nIter = NULL, nwarmup = NULL,
                         data_dir = data_savedir, code_dir = code_savedir, 
                         savedir = stanfit_savedir,
                         adapt = 0.95, treedepth = 10, nThinn = 1) {
    
    #******************************************************#
    #                  Step 1: Preparation                 #
    #******************************************************#
    if (!dir.exists(savedir)) {
      dir.create(savedir, recursive = TRUE)
    }  
    
    L <- list()

    dataList <- prep_data4stan(df, modelStr)
    poi      <- create_pois(modelStr)
    model_name <- modelStr
    
    #******************************************************#
    #               Step 2: model estimate                 #
    #******************************************************#
    
    modelFile <- paste0(code_dir, "models_stan/", modelStr,".stan")
    
    if (test == T) {
        options(mc.cores = 1)
        nIter     <- 20
        nChains   <- 1 
        nWarmup   <- floor(nIter/2)
        nThin     <- 1
        suffix = paste("_", as.character(nIter), "_test", sep = "")
        
    } else {
        options(mc.cores = 4)
        nIter     <- nIter
        nChains   <- 4 
        if (is.null(nwarmup)) {
            nWarmup <- floor(nIter/2)
        } else {
            nWarmup = nwarmup
        }
        nThin     <- nThinn
        suffix = paste("_", as.character(nIter), sep = "")
    }
    
    cat("Estimating", modelStr, "model... \n")
    cat("Variant: ", modelStr, suffix, " ... \n", sep = "")
    startTime = Sys.time(); print(startTime)
    cat("Calling", nChains, "simulations in Stan... \n")
    
    rstan_options(auto_write = TRUE)
    
    fit = stan(file = modelFile, 
               data = dataList,
               pars = poi,
               chains  = nChains,
               iter    = nIter,
               warmup  = nWarmup, 
               thin    = nThin,
               # init    = "random", # default option is "random"
               # seed    = 1362405440,
               control = list(adapt_delta = adapt, max_treedepth = treedepth),
               # open_progress = T,
               verbose = FALSE) 
    
    cat("Finishing", modelStr, "model simulation ... \n")
    endTime = Sys.time(); print(endTime)  
    cat("It took", as.numeric(difftime(endTime, startTime, units = "secs")), "seconds\n")
    
    L$data <- dataList
    L$fit  <- fit
    
    if (saveFit == TRUE) {
      saveRDS(L, file = paste0(savedir, model_name, suffix, '.RData'))
    }
    
    return(L)
} # end of main function

# other functions #
prep_data4stan <- function(df, modelstr){
  
    #%%%%%%%%%%% Read-in and clean the data %%%%%%%%%%%%%#  
    library(DescTools) # only available for R 4.0
    
    sbjcount <- as.numeric(factor(df$participant))
    df <- mutate(df, sbjcount)
    sbjs  <- unique(sbjcount)
    sbjno <- length(sbjs)
    
    #%%%%%%%%%%% Re-format the data %%%%%%%%%%%%%#
    # This is recommended by hBayesDM package
    Tsbj <- as.vector(rep(0, sbjno)) # number of valid trials for each subject
    for (i in 1:sbjno)  {
      curSbj  <- sbjs[i]
      Tsbj[i] <- sum(sbjcount == curSbj)  # Tsbj[N]
    }
    
    # Setting maxTrials
    maxTrials <- max(Tsbj)
    
    # Information for user continued
    cat(" # of (max) trials per subject = ", maxTrials, "\n\n")
    
    validTrialNo <- array(0, c(sbjno, maxTrials))
    q4 <- array(0, c(sbjno, maxTrials))
    Delay <- array(0, c(sbjno, maxTrials))
    Ineq <- array(0, c(sbjno, maxTrials))
    Eff_keep <- array(0, c(sbjno, maxTrials))
    Eff_alter <- array(0, c(sbjno, maxTrials))
    Condition <- array(0, c(sbjno, maxTrials))
    
    
    dataList <- list()
    
    # m <- as.numeric(substr(modelstr,11,11))
    
    for (i in 1:sbjno) {
      curSbj       <- sbjs[i]
      useTrials    <- Tsbj[i]
      tmp          <- subset(df, df$sbjcount == curSbj)
      validTrialNo[i, 1:useTrials] <- tmp$TrialNo_stan
      
      Ineq[i, 1:useTrials] <- tmp$gini
      Condition[i, 1:useTrials] <- as.numeric(tmp$type_idx)
      Eff_keep[i, 1:useTrials] <- tmp$s.eff_keep
      Eff_alter[i, 1:useTrials] <- tmp$s.eff_alter
      
      if (stringr::str_detect(modelstr, "month") == 1) {
        Delay[i, 1:useTrials] <- tmp$Delay_month
      } else if (stringr::str_detect(modelstr, "year") == 1) {
        Delay[i, 1:useTrials] <- tmp$Delay_year
      } else {
        Delay[i, 1:useTrials] <- tmp$Delay_num
      }
      
      if (stringr::str_detect(modelstr, "z") == 1) {
        q4[i, 1:useTrials] <- tmp$z.q4
      } else if (stringr::str_detect(modelstr, "rescale") == 1) {
        q4[i, 1:useTrials] <- tmp$q4_modi
      } else {
        q4[i, 1:useTrials] <- tmp$q4
      }
      
    }
    
    dataList <- list(
      S             = sbjno,
      T             = maxTrials,
      Tsbj          = Tsbj,
      validTrialNo  = validTrialNo,
      q4            = q4,
      Delay         = Delay,
      Ineq          = Ineq,
      Eff_keep      = Eff_keep,
      Eff_alter     = Eff_alter,
      Group         = Condition
    )
    
    return(dataList)
  
} 

create_pois <- function(model){
  
  pois <- list() # pois: parameters of interest
  
  if (substr(model,1,23) == 'DJ_all_rescale_m1-4_hbm') {
    pois <- c("gamma_help","gamma_punish","k_help","k_punish","w0","w1","sig",
              "mu_gamma_help","mu_gamma_punish","mu_k_help","mu_k_punish",
              "mu_w0","mu_w1","mu_sig")

  }

  pois <- c(pois,"log_lik","U_alter","U_keep","Rating_pred")
  
  return(pois)
}

