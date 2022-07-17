#Ordinal Linear Regression (OLR) and GAM Ordinal Regression (GAM OR) to predict support for Capitol Riots
# See "Christian Nationalism and Political Violence: Victimhood, Racial Identity, Conspiracy, 
# and Support for the Capitol Attacks" (Armaly et al. (2021))

library(MASS)
library(mgcv) #install.packages("mgcv")

#Toy example to show efficacy of GAM OR over GLM OLR 
# in dataset with nonlinear latent variable relation Z = a + f(x) + \epsilon
#Generate data
n = 1000
dat = data.frame(x1 = runif(n,-1,1), x2=4*pi*runif(n)) 
#introduce periodicity for sin() to yield more nonlinear latent pattern 
dat$f = dat$x1^2 + sin(dat$x2)
dat$y_latent = dat$f + rnorm(n,dat$f) 
dat$y = ifelse(dat$y_latent<0,1, ifelse(dat$y_latent<0.5,2,3)) 

#Train/test split
test_size <- 0.20
indices <- sample(n)
train_indices <- indices[c(1:(n*(1-test_size)))]
train_dat <- dat[train_indices,]
test_dat <- dat[-train_indices,]
test_y <- dat$y[-train_indices]

#GLM OLR
polr_model = polr(as.factor(y)~x1 + x2, data=train_dat)
preds <- unclass(predict(polr_model, test_dat))
acc <- sum(test_y - preds == 0) / length(test_y)
acc #test acc
table(preds, test_y) #test preds
polr_model$zeta #cut points

#GAM OR
ocat_model = gam(y~s(x1)+s(x2), family=ocat(R=3), data=train_dat) 
plot(ocat_model, page=1)
preds <- max.col(predict(ocat_model,test_dat,type="response",se=TRUE)$fit)
acc <- sum(test_y - preds == 0) / length(test_y)
acc #test acc
table(preds, test_y) #test preds
ocat_model$family$getTheta(TRUE) #cut points

##########################################################################################
#Apply to real dataset
library(foreign)
df <- read.dta("Clean Data, Feb 2021.dta")
#Drop responses with NA
df <- df[!is.na(df$justified),]

#Convert to different response partitions
#(base), not alteration to response
R <- 5
#(i)
#df$justified <- ifelse(df$justified == 1, 1, ifelse(df$justified <= 3, 2, 3))
#R<- 3
#(ii)
#df <- df[df$justified != 1,] #drop responses with value 1 to yield 2,..,5, R=4 categories
#R <- 4
features <- c("victimhood2","whiteidentity2", "nationalism2") #features of primary interest

#Train/test split
test_size <- 0.20
n <- length(df)
indices <- sample(n)
train_indices <- indices[c(1:(n*(1-test_size)))]
train_dat <- df[train_indices,]
test_dat <- df[-train_indices,]
test_y <- unclass(df$justified[-train_indices])

#Linear models (built with different features)
lm1<- lm(justified ~ pid2 + ideo2 + evangelical + protestant + catholic +
           otherchristian + otherreligion + attend2 + edu2 + age2 +
           latinx + female + south, data=train_dat)

lm2 <- lm(justified~ victimhood2 + whiteidentity2 + nationalism2
          + pid2 + ideo2 + evangelical + protestant + catholic +
            otherchristian + otherreligion + attend2 + edu2 + age2 +
            latinx + female + south, data=train_dat)

#GLM OLR
polr0 <- polr(as.factor(justified) ~ victimhood2 + whiteidentity2 + nationalism2, data=train_dat)

polr1 <- polr(as.factor(justified) ~ pid2 + ideo2 + evangelical + protestant + catholic +
                otherchristian + otherreligion + attend2 + edu2 + age2 +
                latinx + female + south, data=train_dat, Hess=TRUE)

polr2 <- polr(as.factor(justified) ~ victimhood2 + whiteidentity2 + nationalism2 + pid2 +
                ideo2 + evangelical + protestant + catholic + otherchristian +
                otherreligion + attend2 + edu2 + age2 + latinx + female +
                south, data=train_dat, Hess=TRUE)

preds <- unclass(predict(polr0, test_dat[features])) #if many features used to train, may yield NAs
diff <- (test_y - preds == 0)
acc <- sum(diff[!is.na(diff)]) / length(diff[!is.na(diff)])
acc #test acc
table(preds, test_y) #test preds
polr0$zeta #cut points

#chi square statistic for goodness of fit (GLM OLR)
# 1-pchisq(deviance(polr2),df.residual(polr2)) 

#GAM OLR
#Train with subset of features
ocat_model <- gam(unclass(justified)~ s(victimhood2) + s(whiteidentity2) + s(nationalism2),
                 family=ocat(R=R),data=train_dat)

plot(ocat_model,page=1)

preds <- max.col(predict(ocat_model,test_dat[features],type="response",se=TRUE)$fit)
acc <- sum(test_y - preds == 0) / length(test_y)
acc #test acc
table(preds, test_y) #test preds
ocat_model$family$getTheta(TRUE) #cut points

