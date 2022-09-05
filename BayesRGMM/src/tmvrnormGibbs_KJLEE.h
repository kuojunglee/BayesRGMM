//
//  tmvrnormGibbs.h
//  
//
//  Created by kuojung on 2020/3/11.
//
#include <RcppArmadillo.h>
#include <RcppDist.h>
//#include <RcppDist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

using namespace std;

using namespace Rcpp;
using namespace arma;

#ifndef tmvrnormGibbs_KJLEE_h
#define tmvrnormGibbs_KJLEE_h

mat rtmvnorm_gibbs_KJLEE(int n, vec mean, mat Sigma, vec lower, vec upper, int burn_in, vec start_value, int thinning);

vec dmvnorm_arma(arma::mat const &x,
                       arma::vec const &mean,
                       arma::mat const &sigma,
                 bool const logd);
double cpp_tmvnorm_prob(vec l, vec u, vec m, mat S, int N); 
#endif /* tmvrnormGibbs_h */
