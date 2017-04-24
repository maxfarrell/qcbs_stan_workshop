##########################
##  QCBS Stan Workshop  ##
##########################

# An introduction to Stan, rstan & rstanarm

# Maxwell J. Farrell
# April 24th 2017

###############
## Libraries ##
###############
require(rstan)
require(gdata)
require(ggplot2)
require(bayesplot)
require(rstanarm)
require(shinystan)
require(loo)
###############

# Remember! Bayesian model building is not always easy!
# It can require a lot of work to build up to a good working model
# that works for your data and also reflects what you 
# know about the world.

# First, with any model building get to know your data.
# Explore it, plot it, calculate those summary statistics.

# Once you have a sense of your data and what you want to learn,  
# you enter the iterative process of building a Bayesian model:

# 1.  Choose your model
# 2.  Simulate data & make sure your model is doing what you think it's doing 
# 2.  Pick priors (informative? not? do you have external data you could turn into a prior?)
# 3.  Inspect model convergence (traceplots, rhats, and no divergent transitions...)
# 4.  Generate posterior predictions and check how they compare to your data!
# 5.  Repeat...


#####################
## Simulating data ##
#####################

# To start we will simulate data under a linear model:
# y = alpha + beta*x + error

set.seed(420)

N <- 100
alpha <- 2.5
beta <- 0.2
sigma <- 6
x <- rnorm(N, 100, 10)
y <- rnorm(N, alpha + beta*x, sigma)

# This ^ is equivalent to the following:
# y <- alpha + beta*x + rnorm(N, 0, sigma)

# Have a look
plot(y~x, pch=20)

# Plot the "true" relationship
abline(alpha, beta, col=4, lty=2, lw=2)

# Run an "lm" model for comparison
lm1 <- lm(y ~ x)
summary(lm1)
abline(lm1, col=2, lty=2, lw=3)


#####################
## STAN REFERENCES ##
#####################

# WEBSITE: http://mc-stan.org/
# MANUAL(v2.14): https://github.com/stan-dev/stan/releases/download/v2.14.0/stan-reference-2.14.0.pdf
# RSTAN: https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html
# STANCON 2017 Intro Course Materials: https://t.co/6d3omvBkrd
# MAILING LIST: https://groups.google.com/forum/#!forum/stan-users

############################
## Our first Stan program ##
############################

# We're going to start by writing a linear model in the 
# language Stan. This can be written in your R script, 
# or saved seprately as a .stan file and called into R.

# A stan program has three required "blocks":

# 1. "data" block: where you declare the data types, 
# their dimensions, any restrictions 
# (i.e. upper= or lower- , which act as checks for Stan),
# and their names.

# Any names will be what you give to your Stan 
# program and also be the names used in other blocks.

# 2. "parameters" block: This is where you indicate the 
# parameters you want to model, their dimensions, restricitons, and name. 
# For a linear regression, we will want to model the intercept, 
# any slopes, and the standard deviation of the errors 
# around the regression line.

# 3. "model" block: This is where you include any sampling statements,
# including the "likelihood" (model) you are using. 
# The model block is where you indicate any prior distributions 
# you want to include for your parameters. If no prior is defined, 
# Stan uses default priors of uniform(-infinity, +infinity). 
# You can restrict priors using restrictions when 
# declaring the parameters (i.e. <lower=0> to make sure a parameter is + )

# Sampling is indicated by the "~" symbol, and Stan already includes
# many common distributions as vectorized functions. 
# See the manual for a comprehensive list.

# There are also optional blocks:
# functions {}
# transformed data {}
# transformed parameters {}
# generated quantitied {}

# Comments are indicated by "//" and are ignored by Stan.

stan_lm1_model <- "
// Stan model for simple linear regression

data {
  int<lower=1> N;       // Sample size
  vector[N] x;          // Predictor
  vector[N] y;          // Outcome
}
parameters {
  real alpha;           // Intercept
  real beta;            // Slope (regression coefficients)
  real<lower=0> sigma;  // Error SD
}
model {
  y ~ normal(alpha + x * beta , sigma);  
//  y ~ alpha + x * beta + normal(0 , sigma);  // Equivalent to above
}
generated quantities {
}
"

# Here we are implicitly using uniform(-infinity, +infinity) priors  
# for our parameters. 

# Sampling

# Stan programs are complied to C++ before being used. For this you must have
# a C++ complier installed (see: https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started)
# If we wanted to call a .stan file, we use the argument file= instead of model_code.
# You can use your model many times per session once you compile it, 
# but you must re-complile when you start a new R session.
model1 <- stan_model(model_code=stan_lm1_model)

# There are many C++ compilers and are often different across systems.
# If your model spits out a bunch of errors / junk, don't worry.
# As long as your model can be used with the sampling() function, it compiled correctly.

# We need to pass the data we want to use to stan as a list of named objects. 
# The names given here need to match the variable names used in the data block.
stan_data <- list(N=N, x=x, y=y)

# We fit our model by using the sampling() function, and providing it with the
# model, the data, and indicating the number of iterations for warmup,
# the total number of iterations, how many chains we want to run, 
# the number of cores we want to use (stan is set up for parallelization), which 
# indicates how many chains are run simultaneously, and the thinning, 
# which is how frequently we want to store our post-warmup iterations.
# thin=1 will keep every iteration, thin-2 will keep every second, etc... 

# Stan automatically uses half of the iterations as warm-up, if 
# the warmup= argument is not specified. 

fit <- sampling(model1, data=stan_data, warmup=500, iter=1000, chains=4, cores=2, thin=1)

################################################
## Accessing the contents of a stanfit object ##
################################################
# https://cran.r-project.org/web/packages/rstan/vignettes/stanfit-objects.html

# Results from sampling() are saved as a "stanfit" object (S4 class)
# We can get summary statistics for parameter estimates, 
# and sampler diagnostics by executing the name of the object:
fit

# From this output we can quickly assess model convergence by 
# looking at the Rhat values for each parameter. 
# When these are at or near 1, the chains have converged.
# There are many other diagnostics, but this is an important one for Stan.

# We can also look at the full posterior of our parameters by extracting
# them from the model object. There are many ways to access the posterior.
posterior <- extract(fit)
str(posterior)
# extract() pulls puts the posteriors for each parameter into a list

# Let's compare to our previous estimate with "lm" 
abline( mean(posterior$alpha), mean(posterior$beta), col=6, lw=2)
# The result is identical to the lm output. 
# This is because we are using a simple model, and have put
# uniform priors on our parameters. 


# One way to visualize the variability in our estimation of the regression line
# is to plot multiple estimates from the posterior:

# Plotting the posterior distribution of regression lines
for (i in 1:500) {
  abline(posterior$alpha[i], posterior$beta[i], col="gray", lty=1)
}
# Let's plot the "true" regression line on top:
abline(alpha, beta, col=4, lty=2, lw=3)


# Let's try again, but now with "strong" priors on the relationship
# Try changing the priors yourself and see what happens!

stan_lm1_model_strongPriors <- "
// Stan model for simple linear regression

data {
  int<lower=1> N;       // Sample size
  vector[N] x;          // Predictor
  vector[N] y;          // Outcome
}
parameters {
  real alpha;           // Intercept
  real beta;            // Slope (regression coefficients)
  real<lower=0> sigma;  // Error SD
}
model {

  alpha ~ normal(10, 0.1);
  beta ~ normal(1, 0.1);

  y ~ normal(alpha + x * beta , sigma);  
//  y ~ alpha + x * beta + normal(0 , sigma);  

}
generated quantities {
}
"
model2 <- stan_model(model_code=stan_lm1_model_strongPriors)
fit2 <- sampling(model2, data=stan_data, warmup=500, iter=1000, chains=4, cores=2, thin=1)

posterior2 <- extract(fit2)
abline( mean(posterior2$alpha), mean(posterior2$beta), col=7, lw=2)
# and compared to the uniform priors:
abline( mean(posterior$alpha), mean(posterior$beta), col=6, lw=3)


#############################
## Convergence Diagnostics ##
#############################

# Before we go on we should check again the Rhat values, the 
# effective sample size (n_eff), and the traceplots of our model 
# parameters to make sure the model has converged and is trustable.

# n_eff is a crude measure of the effective sample size.
# You usually only need to worry is this number is less than 1/100th
# or 1/1000th of your number of iterations. 
# 'Anything over an n_eff of 100 is usually "fine"' - Bob Carpenter

# For traceplots, we can view them directly from the posterior:
par(mfrow=c(3,1))
plot(posterior$alpha, type="l")
plot(posterior$beta, type="l")
plot(posterior$sigma, type="l")

# For simpler models, convergence is usually not a problem unless you 
# run your sampler for too few iterations:
# Try running a model for only 50 iterations and check the traceplots. 
fit_bad <- sampling(model1, data=stan_data, warmup=25, iter=50, chains=4, cores=2, thin=1)
# This also has some "divergent transitions" after warmup, indicating a mis-specified model, 
# or a sampler that has failed to fully sample the posterior. 
posterior_bad <- extract(fit_bad)
plot(posterior_bad$alpha, type="l")
plot(posterior_bad$beta, type="l")
plot(posterior_bad$sigma, type="l")


# We can also get summaries of the parameters through the posterior directly
# We can plot the simluted values to make sure our model is doing what we think it is...
par(mfrow=c(1,3))
plot(density(posterior$alpha), main="Alpha")
abline(v=alpha, col=4, lty=2)
plot(density(posterior$beta), main="Beta")
abline(v=beta, col=4, lty=2)
plot(density(posterior$sigma), main="Sigma")
abline(v=sigma, col=4, lty=2)

# And from the posterior we can direcly calculate the probability 
# of any parameter being over or under a certain value of interest

# Probablility that beta is >0:
sum(posterior$beta>0)/length(posterior$beta)

# Probablility that beta is >0.2:
sum(posterior$beta>0.2)/length(posterior$beta)


# While we can work with the posterior directly, 
# rstan has a lot of these functions built-in as ggplot objects
traceplot(fit)
# This is a wrapper for "stan_trace()""
# stan_trace(fit)
# This is much better than our previous plot because 
# it displays each chain overlayed.

# We can also look at the posterior densitites ^ histograms
stan_dens(fit)
stan_hist(fit)

# And we can generate plots which indicate the mean parameter estimates
# and any credible intervals we may be interested in
plot(fit, show_density=FALSE, ci_level=0.5, outer_level=0.95, fill_color="salmon")



#################################
## Posterior Predictive Checks ##
#################################
# Vignettes: https://cran.r-project.org/web/packages/bayesplot/index.html

# For prediction and as another form of model diagnostic, 
# Stan can use random number generators to generate predicted values 
# for each data point, at each iteration.

# This way we can generate predictions that also represent 
# the uncertainties in our model and our data generation process.

# We generate these using the Generated Quantities block. 
# This block can be used to get any other information 
# we want about the posterior, or make predictions for new data.

stan_lm1_model_GQ <- "
// Stan model for simple linear regression

data {
  int<lower=1> N;       // Sample size
  vector[N] x;          // Predictor
  vector[N] y;          // Outcome
}
parameters {
  real alpha;           // Intercept
  real beta;            // Slope (regression coefficients)
  real<lower=0> sigma;  // Error SD
}
model {

  y ~ normal(x * beta + alpha, sigma);  

}
generated quantities {
  real y_rep[N];

  for (n in 1:N) {
        y_rep[n] = normal_rng(x[n] * beta + alpha, sigma);
      }
}
"
# Note that vectorization is not supported in the GQ block, 
# so we'll have to put it in a loop, but because this is a C++ program, 
# loops are actually quite fast and the GQ block is only evaluated 
# once per iteration, so it won't add too much time to your sampling.

# Typically, the data generating functions will be the distributions you 
# used in the model block but with an "_rng" suffix. 
# Double-check in the Stan manual to see which rng functions are included.

model3 <- stan_model(model_code=stan_lm1_model_GQ)
fit3 <- sampling(model3, data=stan_data, iter=1000, chains=4, cores=2, thin=1)

# Extracting the y_rep values from posterior
y_rep <- as.matrix(fit3, pars = "y_rep")
# dim(y_rep) # each row is an iteration (single posterior estimate) from the model.

# We can use the "bayesplot" package to make some nicer looking plots.
# This package is a wrapper for many common ggplot2 plots, 
# and has a lot of built-in functions to work with posterior predictions.

# Comparing density of y with densities of y over 200 posterior draws
ppc_dens_overlay(y, y_rep[1:200, ])
# Here we see data (dark blue) fit well with our posterior predictions.

# We can also use this to get compare estimates of summary statistics
ppc_stat(y = y, yrep = y_rep, stat="mean")
# We can change the function passed to "stat" function, and even write our own!

# We can investigate mean posterior prediction per datapoint 
# vs the observed value for each datapoint (default line is 1:1)
ppc_scatter_avg(y = y, yrep = y_rep)

# Here is a list of currently available plots (bayesplot 1.2)
available_ppc()

# You can change the colour scheme in bayesplot too:
color_scheme_view(c("blue", "gray", "green", "pink", "purple",
  "red","teal","yellow"))
# And you can even mix them:
color_scheme_view("mix-blue-red")

# You can set color schemes with:
color_scheme_set("blue")


##################################
## Trying with some "real" data ##
##################################

# Data from Akinyi et al 2013 (Animal Behaviour):
# Role of grooming in reducing tick load in wild baboons (Papio cynocephalus)
# http://datadryad.org/resource/doi:10.5061/dryad.3r8n2

# Data download, load-in and clean-up
# Direct download from Dryad and opening of the .xlsx with the gdata package!
data <- read.xls("http://datadryad.org/bitstream/handle/10255/dryad.44883/Akinyi_etal_2013_AnBeh_dataset1.xlsx?sequence=1", skip=0, method="csv", header=TRUE, stringsAsFactors=FALSE)
names(data) <- tolower(names(data))
names(data)[4] <- "age.years"
names(data)[8] <- "grooming.count"
names(data) <- gsub("[.]at[.]darting","", names(data))

# Only use complete cases
data <- data[complete.cases(data),]

# Recoding sex
data$sex[data$sex=="M"] <- 0
data$sex[data$sex=="F"] <- 1
data$sex <- as.integer(data$sex)

# Getting total tick counts 
data$tick_count <- with(data, adult.tick.count + tick.larvae.count)


####################
## Using rstanarm ##
####################
# Vignette: https://cran.r-project.org/web/packages/rstanarm/vignettes/rstanarm.html

# This is a special extension of rstan that has common regression models pre-compiled.
# Based on "Data Analysis using Regression and Multilevel/Hierarchical Models"
# by Gelman & Hill (2007): http://www.stat.columbia.edu/~gelman/arm/

# Check here for more vignettes: 
# https://cran.r-project.org/web/packages/rstanarm/index.html

# It currently supports building regression models using similar 
# syntax as aov, lm, glm, lmer, glmer, glmer.nb, gamm4, polr, and betareg 

# With this you can easily run most regression models in a Bayesian framework.
# No need to code in Stan, or to compile any models.
# It also comes with wrappers for many rstan and bayesplot functions, 
# functions that can easily update models, or make predictions for new input data.

require(rstanarm)

par(mfrow=c(1,1))

# Trying to re-create figure 3 from the paper
with(data, plot(tick_count ~ grooming.count, ylab="Tick load", xlab="Counts of received grooming"))
abline(lm(tick_count ~ grooming.count, data=data))

# The researchers say that if we are to try to explain variation in tick
# abundances among individuals, we should control for the amount of grooming
# individuals receive (here, amount of grooming over 6 months prior to sampling ticks)

# Model: tick_count ~ grooming.count 
stan_lm1 <- stan_glm(tick_count ~ grooming.count,
                        data = data, family=gaussian, 
                        chains = 4, cores = 2, seed = 15042017)

## Convergence diagnostics for rstanarm objects
color_scheme_set("blue")

# Check convergence diagnostics
plot(stan_lm1, plotfun="trace")
summary(stan_lm1)

## Checking what priors were used
prior_summary(stan_glm1)

# This should be chosen before running the model, but here we will
# just check what default priors rstanarm is using:

# Defaut priors are normal(0,10) for the intercept,
# normal(0,2.5) for the regression coefficient, 
# and half-cauchy(0, 5) for sigma (normal error term)

# This may seem fairly informative, but rstanarm
# automatically centers and scales your variables
# before fitting your model, so these are not as
# informative as they would be if the data were left 
# on their input scales!

# You can change the priors by passing additional
# arguments to the stan_glm function
# ex: prior = normal(0,2.5), prior_intercept = normal(0,10)

## Posterior predictive checks
# The pre-compiled models in rstanarm already include a y_rep 
# variable in the generated quantities block. 

# The "ppc_check()" function is just an rstanarm wrapper for 
# the ppc_* functions from the bayesplot package used above
# that automatically extracts "y_rep" 

pp_check(stan_lm1, plotfun = "stat", stat = "mean")
pp_check(stan_lm1, plotfun = "dens_overlay")

# Looks pretty bad! Maybe this is because we are modelling count data
# using normal (gaussian) errors...

# Let's try a poisson regression (better for count data)
# Instead of writing the full model out again, we can use the "update()"
# function to run the same modelbut using the poisson family
stan_glm1 <- update(stan_lm1, family = poisson) 
plot(stan_glm1, plotfun="trace")
summary(stan_glm1)

pp_check(stan_glm1, plotfun = "stat", stat = "mean")
pp_check(stan_glm1, plotfun = "dens_overlay")

# Looks like the model is underestimating instances of zero counts.
# This is a common issue when using the poisson distribution
# as the variance is equal to the mean for poisson distributions.


# ~~~~ Picking a new model ~~~~ #

# Let's try running the model with a negative binomial distrubution
# which is for modelling count data, but estimates mean and 
# variance ("aka reciprocal dispersion") terms separately
stan_glm2 <- update(stan_glm1, family = neg_binomial_2) 

## Check convergence & priors
plot(stan_glm2, plotfun="trace")
summary(stan_glm2)
prior_summary(stan_glm2)

## Posterior Predictive Checks
pp_check(stan_glm2, plotfun = "stat", stat = "mean")
pp_check(stan_glm2, plotfun = "dens_overlay")

# It looks like we are fitting the data better, but is a bit hard to see 
# because one of our posterior estimates had a large number of 0s...

# Lets write our own function to check the proportion of zeros 
# in the data and the posterior "y_rep" (this is automatically calculated)
prop_zero <- function(x) mean(x == 0)
pp_check(stan_glm2, plotfun = "stat", stat = "prop_zero")
# Looks much better!


# ~~~~ Posterior Predictive Checks to inform model building ~~~~ #

# What about additional predictors? What should we include?
# The researchers indicate that in many primates, participation in 
# grooming bouts differs between the sexes and with life history stage.

# Let's look at the posterior mean of tick counts grouped by sex
pp_check(stan_glm2, plotfun="stat_grouped", group=data$sex, stat=mean)

# Looks like our current model is underpredicting the tick count for males (0), 
# and overpredicting the tick count for females (1)

# Model: tick_count ~ grooming.count + sex + age.years
stan_glm3 <- stan_glm(tick_count ~ grooming.count + sex,
                        data = data, family=neg_binomial_2, 
                        chains = 4, cores = 2, seed = 15042017)


## Check convergence & priors
plot(stan_glm3, plotfun="trace")
summary(stan_glm3)

## Posterior Predictive Checks
pp_check(stan_glm3, plotfun="stat_grouped", group=data$sex, stat=mean)
# looks better!

# What about age?
pp_check(stan_glm3, plotfun="intervals", x=data$age.years, prob=0.95)
# Looks like observed tick count slightly increases with age.
# The observations per age fall within the mean 95% credible intervals 
# according to age, but we maybe we could get better estimates by including it...

# Model: tick_count ~ grooming.count + sex + age.years
stan_glm4 <- stan_glm(tick_count ~ grooming.count + sex + age.years,
                        data = data, family=neg_binomial_2, 
                        chains = 4, cores = 2, seed = 15042017)

## Check convergence & priors
plot(stan_glm4, plotfun="trace")
summary(stan_glm4)
# looks like age is not adding too much to the model
pp_check(stan_glm4, plotfun="intervals", x=data$age.years, prob=0.9)
# Looks like age has a weak effect, but we have reduced variation
# in the our posterior predictions for younger baboons.

# We can visualize the our parameter estimates with the "areas" plot function
plot(stan_glm4, plotfun="areas", prob = 0.95, prob_outer = 1.00)


#############################################
## Interactive Diagnostics with shinystan! ##
#############################################

require(shinystan)

# With the help of shiny, you can also look at your model diagnostics 
# and posterior predictive checks interactively through the 
# shinystan package, which offers many more mcmc diagnostics to explore. 
launch_shinystan(stan_glm4)

# Looks like grooming.count still doesn't have much of an effect at all
# Let's run one more model without it, then compare...

# Model: tick_count ~ sex + age.years 
stan_glm5 <- stan_glm(tick_count ~ sex + age.years,
                        data = data, family=neg_binomial_2, 
                        chains = 4, cores = 2, seed = 15042017)

######################
## Model Comparison ##
######################

# There are many ways to compare models, and in the end it depends on 
# what your initial goal was in building the model, what criteria you
# want to use to define the "best" model, and what kinds of models you are fitting.
# One philosophy is that you should build the model you beleive represents 
# your system of interest well, and there is no need to use formal tests to 
# select a "best" from a variety of candidate models.
# Another philosophy, well phrased by George Box is 
# "All models are wrong but some are useful". In this frame of mind, we 
# could run a series of models, and average across the outcomes of each,
# thereby incorporating uncertainty in model selection.  

# Popular in ecology is the information theoretic approach, often
# employing some flavour of the Aikaike Information Criterion (AIC). 
# You could calculate AIC for Bayesian models, and some statisticians 
# have developed alternative versions such as BIC that includes 
# penalization based on sample size, or DIC, which 
# is often used for hierarchical models. 

# For the Stan development team, they do not promote AIC, BIC, or DIC 
# because they are based on point estimates of the likelihood, and therefore
# are not fully Bayesian. Instead they promote the use of 
# WAIC (Widely applicable information criterion), and 
# LOO (leave-one-out cross-validation).
# These use calculations of the log-likelihood across the entire posterior.

# For predictive models, a common method for picking models is to test
# the predictive accuracy of your model on new data, or some held-out portion of 
# the data you already have (but not the portion you used to build the model).

# If your model does well predicting a random datapoint excluded from your model
# when you build it, it is likely not overfit. However, re-running your model 
# many times, dropping a different data point every time would take a lot of 
# computation time, and be unpractical for complex models.  

# The "loo" package in R employs a algorithm to approximate what the performance
# of your model would be if you performed full leave-one-out cross-validation.

# If you are interested in how the approximation is calculated, 
# and how to determine if you can reasonably make this approximation 
# for you model, please read the original paper by Aki Vehtari, which 
# explains WAIC and LOO: https://arxiv.org/abs/1507.04544
# and check out the example on CRAN: https://cran.r-project.org/web/packages/loo/index.html

##############################################################################
## Model selection based on approximation to leave-one-out cross-validation ##
##############################################################################
# Frequentists would test the null hypothesis that the coefficient on 
# a particular term is zero (like grooming.count seems to be). 
# Bayesians might ask whether such a model is expected to produce better 
# out-of-sample predictions than a model without it. 

# For rstanarm, the log-likelihood necessary for calculated is already
# included in the fit object, but if using your own stan program you 
# need to include it's calculation in the generated quantities block.

require(loo)
loo1 <- loo(stan_glm1)
# The pareto k diangnostic is used to determine if it is appropriate to 
# use the loo approximation, and it is calculated per observation. 
# k values over 0.7 indicate the observation is and outlier and may be 
# problematic, meaning it may have unexpectedly high influence 
# when building your model. 

# We can print the loo() output and visualize the k estimates:
loo1
plot(loo1)

# From Jonah Gabry and Ben Goodrich (rstanarm developers):
# "LOO has the same purpose as the AIC that is used by frequentists. 
# Both are intended to estimate the expected log predicted density (ELPD) 
# for a new dataset. However, the AIC ignores priors and assumes that the 
# posterior distribution is multivariate normal, whereas the functions 
# from the loo package used here do not assume that the posterior distribution 
# is multivariate normal and integrate over uncertainty in the parameters. 
# This only assumes that any one observation can be omitted without having 
# a major effect on the posterior distribution, which can be judged using the plots above."

# One or two moderate outliers (k>0.5) shouldn't have too much of aneffect on the model, 
# Rather than ignoring this, one option is to re-fit the model without these problematic 
# observations, and calculate the loo statistic directly for them.
loo1 <- loo(stan_glm1, k_threshold=0.7)
loo2 <- loo(stan_glm2, k_threshold=0.7)
loo3 <- loo(stan_glm3, k_threshold=0.7)
loo4 <- loo(stan_glm4, k_threshold=0.7)
loo5 <- loo(stan_glm5, k_threshold=0.7)

# We can then compare models:
compare(loo1, loo2, loo3, loo4, loo5)
# Like AIC, lower "looic" values indicate preference over other models,

# Remember! Bayesian model building is not always easy!
# It can require a lot of work to build up to a good working model
# that also reflects what you know about the world.

# The general process is iterative:

# 1.  Pick model 
# 2.  Simulate data to make sure your model is doing what you think it's doing 
# 2.  Pick priors (informative? not? do you have external data you could turn into a prior?)
# 3.  Inspect model convergence (traceplots, rhats, an no divergent transitions...)
# 4.  Generate posterior predictions and check that they look like your data!
# 5.  Repeat...

# Stan is a run by a small, but dedicated group of developers. 
# If you are start using Stan, please get involved in the mailing list. 
# It's a great resource for understanding and diagnosing problems with Stan, 
# and by posting problems you encounter you are helping yourself, 
# and giving back to the community.

