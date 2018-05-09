#' SQBreg
#'
#' Do prediction using sequential bagging method with tree based learning algorithm
#' @name SQBreg
#' @param data.train Training dataset
#' @param data.test Testing dataset
#' @param y Numeric response variable
#'
#' @param res Resampling size. Could not be greater than the input data size.
#' @param reps Replicates for the first bagging, default 100
#' @param cores Use multi-cores, default one core, use cores='maxcores' for full use.
#' @param FunKDE Kernel density estimate function, can use different kernel to fit, default logistic kernel.
#' @param control Use in rpart package, rpart.control to tune the control
#' @param SQBalgorithm.1 Use for the first bagging training. Option: lm, CART(default), knnreg.
#' @param SQBalgorithm.2 Use for the last bagging training. Option: lm, CART(default), knnreg.

#' @importFrom rpart rpart
#' @importFrom rpart rpart.control
#' @importFrom parallel mclapply
#' @importFrom epandist repan
#' @import stats
#' @import utils
#' @importFrom caret knnreg
#' @importFrom triangle rtriangle
#' @return Given testing set input, make a regression prediction
#' @references Breiman L., Friedman J. H., Olshen R. A., and Stone, C. J. (1984)
#' \emph{Classification and Regression Trees.}

#' @references Soleymani, M. and Lee S.M.S(2014). Sequential combination of weighted and nonparametric bagging for classification. \emph{Biometrika}, 101, 2, pp. 491--498.

#' @examples
#' data(hills, package="MASS")
#' rt.df <- hills[sample(nrow(hills)),]
#' data.train <- rt.df[1 : (length(rt.df[, 1]) / 2), ]
#' data.test <- rt.df[-(1 : (length(rt.df[, 1]) / 2)),]

#' fit <- SQBreg(data.train = data.train, data.test = data.test, reps = 50, y = "time")
#' fit
#'
#' @export



SQBreg <- function(data.train, data.test, y, res, reps,
                   cores, FunKDE, control, SQBalgorithm.1, SQBalgorithm.2){

  pb <- txtProgressBar(min = 0, max = 4, style = 3)  #

  data.train = data.train[,c(which(colnames(data.train) != y), which(colnames(data.train) == y))]
  data.test = data.test[,c(which(colnames(data.test) != y), which(colnames(data.test) == y))]

  mingzi <- paste("x", 1 : length(data.train), sep = "")
  mingzi[length(mingzi)] <- "y1"
  colnames(data.train) <- mingzi
  colnames(data.test) <- mingzi
  formula <- y1 ~ .

  if (length(data.test) != length(data.train)){
    stop("training and testing sets must have the same predictors and the same response")
  }

  if (missing(y))
    stop("y is the response variable that must exist")
  if (!is.data.frame(data.train) || !is.data.frame(data.test))
    stop("'Input datasets must be data frame")
  if (is.na(data.train) || is.na(data.test))
    stop("NA must be removed from the data")
  if (length(data.train) != length(data.test))
    stop("Unequal column length")

  if (missing(control)){
    control = list(minsplit = 20, cp=0)
  }

  n.predictors = ncol(data.train) - 1
  maxit <- nrow(data.train) # input data size#  V3
  index <- 1 : maxit
  response <- n.predictors + 1  #  The column position of response

  if (missing(res)){
    res = round(maxit / 2)
  }

  if (missing(reps)){
    reps = 10
  }

  if (missing(SQBalgorithm.1)){
    SQBalgorithm.1 = "CART"
  }

  if (missing(SQBalgorithm.2)){
    SQBalgorithm.2 = "CART"
  }

  RegTree <- function(formula, data.train, res, index, SQBalgorithm, control,...) {
    SQBalgorithm = SQBalgorithm.1
    store <- double(maxit)
    for (i in index) {
      j <- index[i]
      delete1.L <- data.train[-j, ]
      subindex1 <- sample((1 : (maxit - 1))[-j], res, replace=F)
      bootstrap.sample1 <- delete1.L[subindex1, ]

      if (SQBalgorithm == "CART"){
        fit1.step2.lm1 <- rpart(formula=formula, data=bootstrap.sample1, method="anova", control=control)
      }
      if (SQBalgorithm == "lm"){
        fit1.step2.lm1 <- lm(formula=formula, data=bootstrap.sample1)
      }
      if (SQBalgorithm == "KNN"){
        fit1.step2.lm1 <- knnreg(formula = formula, data = bootstrap.sample1, k = 3)
      }

      pred.lm <- predict(fit1.step2.lm1, newdata=data.train[j, ])
      store[i] <- pred.lm
    }
    store
  }

  #   1 cutoff
  setTxtProgressBar(pb, 1)

  if (missing(cores) || cores == 1){
    cores = F
    res.replicate <- replicate(reps, RegTree(formula=formula, data.train, res = res, index = index,
                                             SQBalgorithm = SQBalgorithm.1))
  }

  else if (1 <  cores & cores < 1 + getOption("mc.cores", parallel::detectCores())) {
    res.replicate <- mclapply(1 : reps, function(itr) {
      RegTree(formula=formula, data.train, res = res, index = index,
              SQBalgorithm = SQBalgorithm.1)},
      mc.cores = cores)
    res.replicate <- matrix(unlist(res.replicate), ncol = reps)
  }

  else if (cores == "maxcores"){
    cores = getOption("mc.cores", parallel::detectCores())
    res.replicate <- mclapply(1 : reps, function(itr) {
      RegTree(formula=formula, data.train, res = res, index=index,
              SQBalgorithm = SQBalgorithm.1)},
      mc.cores = cores)
    res.replicate <- matrix(unlist(res.replicate), ncol = reps)
  }

  else if (cores > getOption("mc.cores", parallel::detectCores()) || cores < 1 || cores %% 1 != 0){
    stop("The use number of cores is invalid")
  }


  setTxtProgressBar(pb, 2)

  new.reg.lm100 <- replicate(reps, rowMeans(res.replicate[, sample(ncol(res.replicate), res, replace=T)]))

  setTxtProgressBar(pb, 3)
  # norm.generator150 <- matrix(data=0, nrow=length(new.reg.lm100[, 1]), ncol=reps)



  if (missing(FunKDE) || FunKDE == "logistic")
  {
    FunKDE = function(new.reg.lm100, reps) {

      SIGMA <- sqrt(pi ^ 2 / 3)  #
      c <- 1 / sqrt(1 + SIGMA^2)

      meanMatrix <- matrix(rep(rowMeans(new.reg.lm100), reps), ncol = reps)
      sdMatrix <- matrix(rep(apply(new.reg.lm100, 1, sd), reps), ncol = reps)
      Zi <- matrix(rlogis(length(sdMatrix), 0, SIGMA), ncol = reps)  #

      norm.generator150 <- meanMatrix + c * (new.reg.lm100 -  meanMatrix + sdMatrix * Zi)

      return(norm.generator150)
    }
  }

  else if (FunKDE == "gaussian")
  {
    FunKDE = function(new.reg.lm100, reps) {

      SIGMA <- sqrt(1)  #

      c <- 1 / sqrt(1 + SIGMA^2)
      sigma.hat <- apply(new.reg.lm100, 1, sd)

      meanMatrix <- matrix(rep(rowMeans(new.reg.lm100), reps), ncol = reps)
      sdMatrix <- matrix(rep(sigma.hat , reps), ncol = reps)

      Zi <- matrix(rnorm(length(sdMatrix), 0, SIGMA), ncol = reps)  #

      norm.generator150 <- meanMatrix + c * (new.reg.lm100 -  meanMatrix +
                                               sdMatrix * Zi)
      return(norm.generator150)
    }
  }

  else if (FunKDE == "rectangle")
  {
    FunKDE = function(new.reg.lm100, reps) {

      SIGMA <- sqrt(1/3)  #

      c <- 1 / sqrt(1 + SIGMA^2)
      sigma.hat <- apply(new.reg.lm100, 1, sd)

      meanMatrix <- matrix(rep(rowMeans(new.reg.lm100), reps), ncol = reps)
      sdMatrix <- matrix(rep(apply(new.reg.lm100, 1, sd), reps), ncol = reps)

      Zi <- matrix(runif(length(sdMatrix), -1, 1), ncol = reps)  #


      norm.generator150 <- meanMatrix + c * (new.reg.lm100 -  meanMatrix +
                                               sdMatrix * Zi)

      return(norm.generator150)
    }
  }

  else if (FunKDE == "epan")
  {
    FunKDE = function(new.reg.lm100, reps) {

      SIGMA <- sqrt(0.2)  #

      c <- 1 / sqrt(1 + SIGMA^2)

      meanMatrix <- matrix(rep(rowMeans(new.reg.lm100), reps), ncol = reps)
      sdMatrix <- matrix(rep(apply(new.reg.lm100, 1, sd), reps), ncol = reps)

      Zi <- matrix(repan(length(sdMatrix), 0, sqrt(5)), ncol = reps)  #


      norm.generator150 <- meanMatrix + c * (new.reg.lm100 -  meanMatrix +
                                               sdMatrix * Zi)

      return(norm.generator150)
    }
  }

  else if (FunKDE == "triangle")
  {
    FunKDE = function(new.reg.lm100, reps) {

      SIGMA <- sqrt(1/6)  #

      c <- 1 / sqrt(1 + SIGMA^2)

      meanMatrix <- matrix(rep(rowMeans(new.reg.lm100), reps), ncol = reps)
      sdMatrix <- matrix(rep(apply(new.reg.lm100, 1, sd), reps), ncol = reps)

      Zi <- matrix(rtriangle(length(sdMatrix), -1, 1), ncol = reps)  #

      norm.generator150 <- meanMatrix + c * (new.reg.lm100 -  meanMatrix +
                                               sdMatrix * Zi)

      return(norm.generator150)
    }
  }
  else {
    stop("response generator invalid")
  }

  KDE100.generator150 <- FunKDE(new.reg.lm100, reps)
  KDEtraining.newL <- data.frame(data.train[, -length(data.train), drop = FALSE], KDE100.generator150)
  KDEdf.lm <- matrix(0, nrow = nrow(data.test) + nrow(data.train) - maxit, ncol = reps)

  if(SQBalgorithm.2 == "CART"){
    for (i in (n.predictors + 1) : (reps + n.predictors)) {
      gongshi.KDE <- as.formula(paste("KDEtraining.newL[, i] ~",
                                      paste(attr(terms.formula(formula, data = data.train), "term.labels"),
                                            sep = "", collapse = "+")))
      KDEfit.lm <- rpart(formula = gongshi.KDE, data = KDEtraining.newL, method="anova")
      KDEdf.lm[, i - n.predictors] <- predict(KDEfit.lm, newdata=data.test)
    }
  }

  if(SQBalgorithm.2 == "lm"){
    for (i in (n.predictors + 1) : (reps + n.predictors)) {
      gongshi.KDE <- as.formula(paste("KDEtraining.newL[, i] ~",
                                      paste(attr(terms.formula(formula, data = data.train), "term.labels"),
                                            sep = "", collapse = "+")))
      KDEfit.lm <- lm(formula = gongshi.KDE, data = KDEtraining.newL)
      KDEdf.lm[, i - n.predictors] <- predict(KDEfit.lm, newdata=data.test)
    }
  }

  if(SQBalgorithm.2 == "KNN"){
    for (i in (n.predictors + 1) : (reps + n.predictors)) {
      gongshi.KDE <- as.formula(paste("KDEtraining.newL[, i] ~",
                                      paste(attr(terms.formula(formula, data = data.train), "term.labels"),
                                            sep = "", collapse = "+")))
      KDEfit.lm <- knnreg(formula = gongshi.KDE, data = KDEtraining.newL, k = 3)
      KDEdf.lm[, i - n.predictors] <- predict(KDEfit.lm, newdata=data.test)
    }
  }

  #   3 cutoff

  KDEdf.lm <- as.data.frame(KDEdf.lm)
  final.prediction <- rowMeans(KDEdf.lm) #####################  Prediction results for testset.

  #   4 cutoff
  setTxtProgressBar(pb, 4)
  return(final.prediction)
}


