% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CorrMat.HSD.r
\name{CorrMat.HSD}
\alias{CorrMat.HSD}
\title{To compute the correlation matrix in terms of hypersphere decomposition approach}
\usage{
CorrMat.HSD(w, delta)
}
\arguments{
\item{w}{a design matrix is used to model the HS parameters as functions of subject-specific covariates.}

\item{delta}{an \eqn{a \times 1} vector of unknown parameters to model the HS parameters.}
}
\value{
a correlation matrix
}
\description{
The correlation matrix is reparameterized via hyperspherical coordinates angle parameters for \cr
trigonometric functions,
and the angle parameters are referred to hypersphere (HS) parameters. In order to obtain the unconstrained estimation
of angle parameters and to reduce the number of parameters for facilitating the computation,
we model the correlation structures of the responses in terms of the generalized linear models
}
\examples{
\dontrun{
library(BayesRGMM)
rm(list=ls(all=TRUE))
T = 5 #time points
HSD.para = c(-0.5,  -0.3) #the parameters in HSD model
a = length(HSD.para)
w = array(runif(T*T*a), c(T, T, a)) #design matrix in HSD model
signif(CorrMat.HSD(w, HSD.para), 4)
}
}
\references{
{
\insertRef{Zhang:etal:2015}{BayesRGMM}
}
}
\author{
Kuo-Jung Lee \href{mailto:kuojunglee@ncku.edu.tw}{kuojunglee@ncku.edu.tw}
}
