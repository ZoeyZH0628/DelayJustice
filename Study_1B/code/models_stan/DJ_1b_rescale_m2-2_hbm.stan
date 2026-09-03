// DJ_Ineq_allsbjs_rescale_m1_hbm.stan
// Author: Yang Hu
// Established Date: Oct. 17, 2022
// Last Change: Feb. 22, 2023
// Project: Delay Justice (DJ)                                   
// PI: Yang Hu                                                    
// Acknowledgment: "hBayesDM" package 

// This code aims to get parameter estimates and other relevant measures 
// based on the inequality (Ineq) via
// Hierarchical Bayesian Modelling (hbm)

// All models assumes same pars across human and computer conditions
// 2 key pars: gamma (inequality), k (discounting factor)
// Each key par per group
// allsbjs means model all sbjs in a between-group manner
// rescale the original rating: [-50,50] to [0,1]
// namely: Rating_modi = Rating/100 + 0.5

// Model 1 (m1) assumes same w0 (intercept) and sig (variance) across groups

data {
  int <lower=1> S; // number of subjects
  int <lower=1> T; // number of maxtrials among all sbjs
  // int <lower=1> C; // number of conditions
  int <lower=1, upper=T> Tsbj[S]; // numbers of valid trials per sbj
  int <lower=0> validTrialNo[S,T]; // valid trial numbers per sbj
  // int <lower=0,upper=1> IsAlter[S,T]; // 0 for keep, 1 for alter
  real q4[S,T]; // justice rating
  int <lower=0> Delay[S,T]; // time delay
  real <lower=0, upper=1> Ineq[S,T]; // trial-wise gini coef. (difference between A and B's payoff)
  real <lower=0, upper=1> Eff_keep[S,T]; // trial-wise coefficiency (if the pp chooses to keep)
  real <lower=0, upper=1> Eff_alter[S,T]; // trial-wise coefficiency (if the pp chooses to alter)
  real <lower=0, upper=1> Group[S,T]; // help or punish: 0 for help, 1 for punish
  
}

parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters  
  vector[6] mu_pr; // 1: mu_gamma_help, 2: mu_gamma_punish
                   // 3: mu_k,
                   // 4: mu_w0, 5: mu_w1,
                   // 6: mu_sig
  vector<lower=0>[6] sigma; // single sigma_gamma, sigma_k, sigma_w0, sigma_w1, sigma_sig
                            // for both groups
    
  // Subject-level raw parameters (for Mathematics trick)
  vector[S] gamma_help_pr;
  vector[S] gamma_punish_pr;
  vector[S] k_pr;
  vector[S] w0_pr;
  vector[S] w1_pr;
  vector[S] sig_pr;    
}

transformed parameters {
  // Transform subject-level raw parameters 
  vector<lower=-1,upper=1>[S] gamma_help; // inequality aversion  [0,1]
  vector<lower=-1,upper=1>[S] gamma_punish; // inequality aversion  [0,1]
  vector<lower=-1,upper=1>[S] k; // delay discount  [0,1]
  vector<lower=-1,upper=1>[S] w0; // scaling constant  [-1,1]
  vector<lower=0,upper=3>[S] w1; // scaling constant  [-3,3]
  vector<lower=0>[S] sig; // sigma [0,+]
  
  for (i in 1:S) {
    gamma_help[i] = -1 + Phi_approx( mu_pr[1] + sigma[1] * gamma_help_pr[i] )*2;
    gamma_punish[i] = -1 + Phi_approx( mu_pr[2] + sigma[2] * gamma_punish_pr[i] )*2;
    k[i] = Phi_approx( mu_pr[3] + sigma[3] * k_pr[i] );
    w0[i] = -1 + Phi_approx( mu_pr[4] + sigma[4] * w0_pr[i] )*2;
    w1[i] = Phi_approx( mu_pr[5] + sigma[5] * w1_pr[i] )*3;
    sig[i] = exp( mu_pr[6] + sigma[6] * sig_pr[i] );
    // IT[i] = Phi_approx( mu_p[3] + sigma[3] * IT_pr[i] )*5;
  }
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1); 
  // sigma ~ cauchy(0, 5);// Default:; Alternatives: cauchy(0,3); normal(0,2) or normal(0,3)
  sigma ~ normal(0, 2);
    
  // individual parameters
  gamma_help_pr ~ normal(0, 1);
  gamma_punish_pr ~ normal(0, 1);
  k_pr ~ normal(0, 1);
  w0_pr ~ normal(0, 1);
  w1_pr ~ normal(0, 1);
  sig_pr ~ normal(0, 1); 
  
  for (i in 1:S) {
    
    // Define values
    real U_keep; // U_keep is the utility for the keep option
    real U_alter; // U_alter is the utility for the alternative (help or punish) option
    real d_U; // d_U = U_alter - U_keep
      
    for (t in 1:(Tsbj[i])) {
      
      // Calculate utility of both options and their difference
      if (Group[i,t] == 0) { // help Block
      
          U_keep = 1 - gamma_help[i]*Ineq[i,t]*(2 - 1/(1 + k[i]*Delay[i,t])); // 1 stands for complete justice
          U_alter = 1/(1 + k[i]*Delay[i,t]); // discount with time

      } else if (Group[i,t] == 1) { // punish Block
      
        U_keep = 1 - gamma_punish[i]*Ineq[i,t]*(2 - 1/(1 + k[i]*Delay[i,t])); // 1 stands for complete justice
        U_alter = 1/(1 + k[i]*Delay[i,t]); // discount with time
 
      }
      
      d_U = U_alter - U_keep;
      
      // Linking the utility to the justice rating
      q4[i,t] ~ normal(w0[i] + d_U * w1[i], sig[i]);
      
      }// end of trial 
 
    } //end of subjects
} // end of model section

generated quantities {
  // For group level parameters 
  real mu_gamma_help;
  real mu_gamma_punish;
  real mu_k;
  real mu_w0;
  real mu_w1;
  real mu_sig;
  // real<lower=0,upper=5> mu_IT;
  
  // For log-likelihood and utility calculation
  // See above for explanation
  
  real U_keep[S,T];
  real U_alter[S,T]; 
  real d_U;
  real log_lik[S]; // d_U = U_alter - U_keep

  // For posteior predictive check
  real Rating_pred[S,T];
  // real<lower=0,upper=1> percent_Accept[S];
  mu_gamma_help = -1 + Phi_approx(mu_pr[1])*2;
  mu_gamma_punish = -1 + Phi_approx(mu_pr[2])*2;
  mu_k = Phi_approx(mu_pr[3]);
  mu_w0 = -1 + Phi_approx(mu_pr[4])*2;
  mu_w1 = Phi_approx(mu_pr[5])*3;
  mu_sig = exp(mu_pr[6]);
  
  U_alter = rep_array(-999.0,S,T); // Initialize the matrix (concerning the missing trial)
  U_keep = rep_array(-999.0,S,T);
  Rating_pred = rep_array(-999,S,T);
  
  { // local section, this saves time and space
    for (i in 1:S) {
      // Define values
      log_lik[i] = 0;
          
      for (t in 1:(Tsbj[i])) {
    
      // Calculate utility of both options and their difference
      if (Group[i,t] == 0) { // help Block
      
          U_keep[i,t] = 1 - gamma_help[i]*Ineq[i,t]*(2 - 1/(1 + k[i]*Delay[i,t])); // 1 stands for complete justice
          U_alter[i,t] = 1/(1 + k[i]*Delay[i,t]); // discount with time

      } else if (Group[i,t] == 1) { // punish Block
      
        U_keep[i,t] = 1 - gamma_punish[i]*Ineq[i,t]*(2 - 1/(1 + k[i]*Delay[i,t])); // 1 stands for complete justice
        U_alter[i,t] = 1/(1 + k[i]*Delay[i,t]); // discount with time
 
      }
      
      // Linking the utility to the justice rating
      d_U = U_alter[i,t] - U_keep[i,t];
      log_lik[i] = log_lik[i] + normal_lpdf(q4[i,t] | w0[i] + d_U * w1[i], sig[i]);
        
      // Posterior predictive check of rating
      Rating_pred[i,t] = normal_rng(w0[i] + d_U * w1[i], sig[i]);
      
      }// end of trial
      
    }  // end of subjects 
  }   // end of local
   
} // end of section
