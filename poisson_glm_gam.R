
##############################################################################
#Poisson GLM and GAM Implementation in Simulation Studies and Real-world Datasets
##############################################################################
#Poisson GLM (Poisson Loglinear model)
#Simulation study
n <- 100
p <- 10
X <- matrix(rnorm(n*p), nrow=n, ncol=p)
beta <- rnorm(p)

eta <- X %*% beta

link <- function(x) {
  return(log(x))
}

inv_link <- function(x) {
  return(exp(x))
}

grad_link <- function(x) {
  return(1/x)
}

mu <- inv_link(eta) #g(mu) = eta 

y <- matrix(0,n)
for (i in 1:n) {
  y[i] <- rpois(1,mu[i])
}

#IRLS (Slow implementation due to direct call for computation of inverse)
mu_t <- y + 1
iters <- 10

IRLS <- function(X, y, mu_t) {
  eta_t <- link(mu_t)
  for (t in 1:iters) {
    D_t <- diag(c(mu_t))
    W_t <- mu_t
    z_t <- eta_t + solve(D_t) %*% (y - mu_t)
    lm.fit <- lm(z_t ~ X-1, weights = W_t)
    beta_t <- lm.fit$coefficients
    eta_t <- X %*% beta_t
    mu_t <- inv_link(eta_t)
  }
  
  return(list(beta_t, solve(t(X) %*% diag(c(W_t)) %*% X)))
} 

result <- IRLS(X, y, mu_t)
beta_pred <- result[[1]]
sigma_pred <- result[[2]]

sum((beta_pred - beta)^2) #L2 norm for weights
c(beta_pred[1] - 1.96 * sigma_pred[1,1], beta_pred[1] + 1.96 * sigma_pred[1,1]) #C.I.

#Run on real dataset
library(MASS)
library(mgcv) #install.packages("mgcv")
library(foreign)

df <- read.csv("twitter_analysis.csv")
#Drop responses with NA
df <- df[!is.na(df$fakenewstot),]

features <- c('fakenewstot', 'female','white','income', 'party_01', 'ideology', 'cynicism')
nonlabel_features <- c('female','white','income', 'party_01', 'ideology', 'cynicism')
df <- df[features]

set.seed(42)
n<-dim(df)[1]; test_size <- 0.33
indices <- sample(n)
train_indices <- indices[c(1:(n*(1-test_size)))]
train_dat <- df[train_indices,]
test_dat <- df[-train_indices,]
test_y <- df$fakenewstot[-train_indices]

train_y <- train_dat$fakenewstot
X <- cbind(1,train_dat[nonlabel_features])
X$female <- as.factor(X$female)
X$white <- as.factor(X$white)
X$party_01 <- as.factor(X$party_01)
X <- model.matrix(~., data=X)[,-1]
mu_t <- train_y + 1

result <- IRLS(X, train_y, mu_t) #This takes a bit to run - slow code implementation
beta_pred <- result[[1]]
sigma_pred <- result[[2]]

glm.fit <- glm(fakenewstot ~ as.factor(female) + as.factor(white)
               + income + as.factor(party_01) + ideology + cynicism,
               data=train_dat, family='poisson') #Compare with glm package

beta <- glm.fit$coefficients
sum((beta_pred - beta)^2) #L2 norm between weights
c(beta_pred[1] - 1.96 * sigma_pred[1,1], beta_pred[1] + 1.96 * sigma_pred[1,1]) #C.I.

summary(glm.fit)
sqrt(sum((exp(predict(glm.fit, test_dat)) - test_y)^2)/length(test_y))

#Ordinary Linear Regression for comparison
beta_ols <- solve(t(X) %*% X) %*% t(X) %*% train_y
sigma_ols <- sum((train_y - X %*% beta_ols)^2)/(length(train_y)-length(beta_ols))
std_err <- sqrt(diag(solve(t(X) %*% X) * sigma_ols))
ub_ols_CI <- beta_ols + 1.96 * std_err
lb_ols_CI <- beta_ols - 1.96 * std_err

lm.fit <- lm(train_y ~ 0 + X)
sum((beta_ols - lm.fit$coefficients)^2) #L2 norm between weights
confint(lm.fit)

X_test <- cbind(1,test_dat[nonlabel_features])
X_test$female <- as.factor(X_test$female)
X_test$white <- as.factor(X_test$white)
X_test$party_01 <- as.factor(X_test$party_01)
X_test <- model.matrix(~., data=X_test)[,-1]
sqrt(sum((X_test %*% beta_ols - test_y)^2)/length(test_y))

##############################################################################
#GAM Implementation
#Simulation study
q1 <- 4 
q2 <- 5
mu_beta <- 100
beta1 <- rnorm(q1, c(mu_beta,-0.5*mu_beta), 0.5)
beta2 <- rnorm(q2, c(mu_beta,-0.5*mu_beta), 0.5)
beta1[1] <- beta1[2] <- 0.01
beta2[1] <- beta2[2] <- 0.01

rk<-function(x,z) { 
  ((z-0.5)^2-1/12)*((x-0.5)^2-1/12)/4 - 
    ((abs(x-z)-0.5)^4 -(abs(x-z)-0.5)^2/2+7/240)/24 
}

xk1 <- c(2/6 , 4/6)
xk2 <- c(1/6, 3/6, 5/6)

n <- 100
x1 <- runif(n)
x2 <- runif(n)

f1_x <- function(x1, xk1) {
  return( beta1[1]*1 + beta1[2]*x1 + beta1[3]*rk(x1, xk1[1]) 
          + beta1[4]*rk(x1, xk1[2]) )
}

f2_x <- function(x2, xk2) {
  return( beta2[1]*1 + beta2[2]*x2 + beta2[3]*rk(x2, xk2[1]) 
          + beta2[4]*rk(x2, xk2[2]) + beta2[5]*rk(x2, xk2[3]))
}

#Plots of true underlying functions 
#Check back here for reference and comparison
plot(x1, f1_x(x1,xk1)) ; title('f1')
plot(x2, f2_x(x2,xk2)) ; title('f2')
xp<-0:100/100 
plot(xp, f1_x(xp,xk1) + f2_x(xp,xk2))
title('Total 2-term additive effect, f1 + f2')

spl.X<-function(x,xk) { 
  q <- length(xk)+2 
  n <- length(x) 
  X <- matrix(1,n,q)
  X[,2] <- x 
  X[,3:q] <- outer(x,xk,FUN=rk) 
  return(X)
  }

#Deterministic Non-additive model
#Regression spline for non-noisy data
xk<-1:4/5 #rank q=6 basis to approximate rank q=4 basis f1(x) 
X<-spl.X(x1,xk) 
lm.fit<-lm(f1_x(x1,xk1)~X-1) 
Xp<-spl.X(xp,xk) 
plot(xp,Xp%*%lm.fit$coefficients, col='blue', xlab='x', ylab='y') # plot fitted spline
lines(xp, f1_x(xp,xk1), col='green')
title('Predicted and true spline (deterministic)')

spl.S<-function(xk) { 
  q <- length(xk)+2
  S<-matrix(0,q,q)
  S[3:q,3:q] <- outer(xk,xk,FUN=rk)
  return(S)
  } 

mat.sqrt<-function(S) { 
  d <- eigen(S,symmetric=TRUE) 
  rS<-d$vectors%*%diag(d$values^0.5)%*%t(d$vectors) 
  return(rS)
  }
  
prs.fit<-function(y,x,xk,lambda) { 
  q <- length(xk)+2 
  n <- length(x) 
  Xa <- rbind(spl.X(x,xk), mat.sqrt(spl.S(xk))*sqrt(lambda)) 
  y[(n+1):(n+q)]<-0 
  return(lm(y~Xa-1))
  }

#Penalized regression spline for noisy data
xk<-1:7/8 #rank q=9 basis to approximate rank q=4 basis f1(x)
y <- f1_x(x1,xk1)+rnorm(n,0,mean(abs(f1_x(x1,xk1)))/10)
prs.model<-prs.fit(y,x1,xk,0.0001) 
xp<-0:100/100
Xp<-spl.X(xp,xk) 
plot(x1,y)
lines(xp,Xp%*%prs.model$coefficients)
title('Predicted spline for noisy data')

#Consider (deterministic) additive model
am.setup<-function(x1,x2,q=10) {
  xk1 <- quantile(unique(x1),1:(q-2)/(q-1)) 
  xk2 <- quantile(unique(x2),1:(q-2)/(q-1)) 
  S <- list()
  S[[1]] <- S[[2]] <- matrix(0,2*q-1,2*q-1) 
  S[[1]][2:q,2:q] <- spl.S(xk1)[-1,-1] 
  S[[2]][(q+1):(2*q-1),(q+1):(2*q-1)] <- spl.S(xk2)[-1,-1] 
  n<-length(x1) 
  X<-matrix(1,n,2*q-1) 
  X[,2:q]<-spl.X(x1,xk1)[,-1] 
  X[,(q+1):(2*q-1)]<-spl.X(x2,xk2)[,-1] 
  return(list(X=X,S=S)) 
  }
  
result <- am.setup(x1,x2,q=10)
X <- result[[1]]
S <- result[[2]]

fit.am <- function(y,X,S,sp){
  rS <- mat.sqrt(sp[1]*S[[1]]+sp[2]*S[[2]]) 
  q <- ncol(X)
  n <- nrow(X) 
  X1 <- rbind(X,rS)
  y1 <- y 
  y1[(n+1):(n+q)] <- 0 
  b <- lm(y1~X1-1) 
  trA <- sum(influence(b)$hat[1:n]) 
  norm <- sum((y-fitted(b)[1:n])^2) 
  return(list(model=b,gcv=norm*n/(n-trA)^2,sp=sp))
  }

y <- f1_x(x1,xk1) + f2_x(x2,xk2) + 
  rnorm(n,0,mean(abs(f1_x(x1,xk1) + f2_x(x2,xk2))))

sp <- c(0,0)
for (i in 1:30) {
  for (j in 1:30) { 
    sp[1] <- 1e-5*2^(i-1); sp[2]<-1e-5*2^(j-1) 
    b <- fit.am(y,X,S,sp) 
    if (i+j==2) best <- b 
    else if (b$gcv<best$gcv) best<-b
    }
  }

best$sp #best smoothing spline parameters

plot(y,fitted(best$model)[1:100], xlab="Predicted y",ylab="True y") 
title('Residual plot')

b <- best$model
b$coefficients[11:19]<-0
f0<-predict(b)
plot(x1,f0[1:100],xlab="x1", ylab=expression(hat(f[1])))
title('First predicted spline in 2-term (Noisy) Additive Model')

##########################################################
#Poisson GAM; y ~ Poiss(lambda), lambda = f1(x) + f2(x)
n <- 100
x1 <- runif(n)
x2 <- runif(n)

y <- matrix(0,n)
for (i in 1:n) {
  y[i] <- rpois(1,exp(f1_x(x1[i],xk1) + f2_x(x2[i],xk2)))
}

result <- am.setup(x1,x2,q=10)
X <- result[[1]]
S <- result[[2]]

fit.gamPoisson <- function(y,X,S,sp) { 
  rS <- mat.sqrt(sp[1]*S[[1]]+sp[2]*S[[2]]) 
  q <- ncol(X) 
  n <- nrow(X) 
  X1 <- rbind(X,rS) 
  b <- rep(0,q)
  b[1] <- 1
  norm <- 0; old.norm <- 1 
  while (abs(norm-old.norm)>1e-2*norm) { 
    eta <- (X1%*%b)[1:n] 
    mu <- exp(eta)
    w <- mu
    z <- (y-mu)/mu + eta 
    z[(n+1):(n+q)] <- 0 
    m <- lm(z~X1-1, weights=c(sqrt(mu),matrix(1,q))) 
    b <- m$coefficients 
    trA <- sum(influence(m)$hat[1:n]) 
    old.norm <- norm; norm <- sum((z-fitted(m))[1:n]^2)
    } 
  return(list(model=m,gcv=norm*n/(n-trA)^2,sp=sp))
  }

sp <- c(0,0)
for (i in 1:30) {
  for (j in 1:30) { 
    sp[1] <- 1e-5*2^(i-1); sp[2]<-1e-5*2^(j-1) 
    b <- fit.gamPoisson(y,X,S,sp) 
    if (i+j==2) best <- b 
    else if (b$gcv<best$gcv) best<-b
  }
}

best$sp

b <- best$model
b$coefficients[11:19]<-0
f0<-predict(b)
plot(x1,f0[1:n],xlab="x2", ylab=expression(hat(f[2])))
title('Predicted 1st spline for Poisson GAM')

b <- best$model
b$coefficients[2:10]<-0
f0<-predict(b)
plot(x2,f0[1:n],xlab="x2", ylab=expression(hat(f[2])))
title('Predicted 2nd spline for Poisson GAM')

gam.fit <- gam(y ~ s(x1) + s(x2) + 1, family='poisson')
plot(gam.fit, pages = 1)

plot(xp, f1_x(xp,xk1)) #True functions
plot(xp, f2_x(xp,xk2))

rS <- mat.sqrt(sp[1]*S[[1]]+sp[2]*S[[2]]) 
X1 <- rbind(X,rS) 
eta <- (X1%*%b$coefficients)[1:n] 
mu <- exp(eta)

#C.I.s (WORK IN PROGRESS - Using Bayesian Approach as in Wood (2006))
vars <- diag(solve(t(X) %*% diag(mu) %*% X + best$sp[1]*S[[1]]+best$sp[2]*S[[2]]))
vars

#Lower bound for 1st function
b <- best$model
lb_b <- b$coefficients - 1.96 * vars
b$coefficients <- lb_b
b$coefficients[11:19]<-0
f1_lb<-predict(b)
plot(x1,f1_lb[1:n],xlab="x1")

#Upper bound for 1st function
b <- best$model
ub_b <- b$coefficients + 1.96 * vars
b$coefficients <- ub_b
b$coefficients[11:19]<-0
f1_ub<-predict(b)
plot(x1,f1_ub[1:n],xlab="x1")

################
#Apply to real-world dataset
rg <- range(train_dat$income) 
train_dat$income <- (train_dat$income- rg[1])/(rg[2]-rg[1]) 

rg <- range(train_dat$cynicism) 
train_dat$cynicism <- (train_dat$cynicism- rg[1])/(rg[2]-rg[1]) 

result <- am.setup(train_dat$income, train_dat$cynicism)
X <- result[[1]]
S <- result[[2]]

sp <- c(0,0)
for (i in 1:30) {
  for (j in 1:30) { 
    sp[1] <- 1e-5*2^(i-1); sp[2]<-1e-5*2^(j-1) 
    b <- fit.gamPoisson(train_y,X,S,sp) 
    if (i+j==2) best <- b 
    else if (b$gcv<best$gcv) best<-b
  }
}

best$sp #best smoothing parameters

#Get plots from fitted coefficients
#Income
b <- best$model; n <- dim(train_dat)[1]
b$coefficients[11:19]<-0
f0<-predict(b)
plot(train_dat$income,f0[1:n],xlab="income", ylab=expression(hat(f[1])))
title('Fake-clicks vs Income')

xp<-1:1000/1000; q<-10
qs <- quantile(unique(train_dat$income),1:(q-2)/(q-1)) 
f_income <- rep(0, 1000); i <- 1; j <- 1
for (beta in b$coefficients[1:10]) {
  if (i == 1) {
    f_income <- f_income + beta * 1
    i <- i + 1
  }
  
  else if (i == 2) {
    f_income <- f_income + beta * xp
    i <- i + 1
  }
  
  else {
    f_income <- f_income + beta * rk(xp, qs[[j]]) #KNOTS
    j <- j + 1
  }
}

plot(xp, f_income, type='l', xlab='Income', ylab=expression(hat(f[1])))
lines(xp, xp*glm.fit$coefficients['income'], lty='dotted')
points(train_dat$income,f0[1:n])
title('Fake-clicks vs Income')

#Cynicism
b <- best$model
b$coefficients[2:10]<-0
f0<-predict(b)
plot(train_dat$cynicism,f0[1:n],xlab="Cynicism", ylab=expression(hat(f[2])))
title('Fake-clicks vs Cynicism')

xp<-1:1000/1000
qs <- quantile(unique(train_dat$cynicism),1:(q-2)/(q-1)) 
f_cynicism <- rep(0, 1000); i <- 1; j <- 1
for (beta in c(1,b$coefficients[11:19])) {
  if (i == 1) {
    f_cynicism <- f_cynicism + beta * 1
    i <- i + 1
  }
  
  else if (i == 2) {
    f_cynicism <- f_cynicism + beta * xp
    i <- i + 1
  }
  
  else {
    f_cynicism <- f_cynicism + beta * rk(xp, qs[[j]]) #KNOTS
    j <- j + 1
  }
}

plot(xp, f_cynicism, type='l', xlab='Cynicism', ylab=expression(hat(f[2])))
lines(xp, xp*glm.fit$coefficients['cynicism'], lty='dotted')
#points(train_dat$cynicism,f0[1:n])
title('Fake-clicks vs Cynicism')
