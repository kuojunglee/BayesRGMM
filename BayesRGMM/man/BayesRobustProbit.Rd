% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/BayesRobustProbit.r
\name{BayesRobustProbit}
\alias{BayesRobustProbit}
\title{Perform MCMC algorithm to generate the posterior samples}
\usage{
BayesRobustProbit(
  fixed,
  data,
  random,
  Robustness = TRUE,
  subset = NULL,
  na.action = "na.exclude",
  arma.order = NULL,
  HS.model = NULL,
  hyper.params = NULL,
  num.of.iter = 20000,
  Interactive = FALSE
)
}
\arguments{
\item{fixed}{a two-sided linear formula object to describe fixed-effects with the response on the left of
a \samp{~} operator and the terms separated by \samp{+} or \samp{*} operators, on the right.
The specification \code{first*second} indicates the cross of \code{first} and \code{second}.
This is the same as \code{first + second + first:second}.}

\item{data}{an optional data frame containing the variables named in \samp{fixed} and \samp{random}.
It requires an ``integer'' variable named by \samp{id} to denote the identifications of subjects.}

\item{random}{a one-sided linear formula object to describe random-effects with the terms separated by
\samp{+} or \samp{*} operators on the right of a \samp{~} operator.}

\item{Robustness}{logical. If 'TRUE' the distribution of random effects is assumed to be \cr
t-distribution; otherwise normal distribution.}

\item{subset}{an optional expression indicating the subset of the rows of \samp{data} that should be used in the fit.
This can be a logical vector, or a numeric vector indicating which observation numbers are to be included,
or a character vector of the row names to be included.  All observations are included by default.}

\item{na.action}{a function that indicates what should happen when the data contain NA’s.
The default action (\samp{na.omit}, inherited from the \samp{factory fresh} value of \cr
\samp{getOption("na.action")}) strips any observations with any missing values in any variables.}

\item{arma.order}{a specification of the order in an ARMA model: the two integer components (p, q) are the AR order and the MA order.}

\item{HS.model}{a specification of the correlation structure in HSD model:
\itemize{
\item \code{HS.model = ~0} denotes independence, that is, \eqn{R_i} is an identity matrix,
\item \code{HS.model = ~IndTime+}\eqn{\cdots}\code{+IndTimer} denotes AR(r) correlation structure,
\item \code{HS.model = ~DiffTime1+}\eqn{\cdots}\code{+DiffTimer} denotes correlation structure related to \eqn{r}th order
of time difference.
}}

\item{hyper.params}{specify the values in hyperparameters in priors.}

\item{num.of.iter}{an integer to specify the total number of iterations; default is 20000.}

\item{Interactive}{logical. If 'TRUE' when the program is being run interactively for progress bar and 'FALSE' otherwise.}
}
\value{
a list of posterior samples, parameters estimates, AIC, BIC, CIC, DIC, MPL, RJR, predicted values,
and the acceptance rates in MH are returned.
}
\description{
This function is used to generate the posterior samples using MCMC algorithm from the probit model
with either the hypersphere decomposition or ARMA models applied to model the correlation structure
in the serial dependence
of repeated responses.
}
\note{
Only a model either HSD (\samp{HS.model}) or ARMA (\samp{arma.order}) model should be specified in the function.
We'll provide the reference for details of the model and the algorithm for performing
model estimation whenever the manuscript is accepted.
}
\examples{
\dontrun{
library(BayesRGMM)
rm(list=ls(all=TRUE))
Fixed.Effs = c(-0.2, -0.3, 0.8, -0.4) #c(-0.2,-0.8, 1.0, -1.2)
P = length(Fixed.Effs) 
q = 1 #number of random effects
T = 5 #time points
N = 100 #number of subjects
num.of.iter = 100 #number of iterations
HSD.para = c(-0.5,  -0.3) #the parameters in HSD model
a = length(HSD.para)
w = array(runif(T*T*a), c(T, T, a)) #design matrix in HSD model

for(time.diff in 1:a)
	w[, , time.diff] = 1*(as.matrix(dist(1:T, 1:T, method="manhattan")) 
 ==time.diff)

#Generate a data with HSD model
 HSD.sim.data = SimulatedDataGenerator(
 Num.of.Obs = N, Num.of.TimePoints = T, Fixed.Effs = Fixed.Effs, 
 Random.Effs = list(Sigma = 0.5*diag(1), df=3), 
Cor.in.DesignMat = 0., Missing = list(Missing.Mechanism = 2, 
 RegCoefs = c(-1.5, 1.2)), Cor.Str = "HSD", 
 HSD.DesignMat.para = list(HSD.para = HSD.para, DesignMat = w))

hyper.params = list(
        sigma2.beta = 1,
        sigma2.delta = 1,
        v.gamma = 5,
        InvWishart.df = 5,
        InvWishart.Lambda = diag(q) )

HSD.output = BayesRobustProbit(
fixed = as.formula(paste("y~-1+", paste0("x", 1:P, collapse="+"))), 
data=HSD.sim.data$sim.data, random = ~ 1, Robustness=TRUE, 
HS.model = ~IndTime1+IndTime2, subset = NULL, na.action='na.exclude', 
hyper.params = hyper.params, num.of.iter = num.of.iter, 
 Interactive=0)
} 
}
\references{
{
\insertRef{Lee:etal:2021}{BayesRGMM}

\insertRef{Lee:etal:2020}{BayesRGMM}

}
}
\author{
Kuo-Jung Lee \href{mailto:kuojunglee@ncku.edu.tw}{kuojunglee@ncku.edu.tw}
}
