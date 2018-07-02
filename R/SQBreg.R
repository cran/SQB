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
#' @param FunKDE Kernel density estimate function. Use different kernel to fit, default logistic kernel.
#' @param control Use in rpart package, rpart.control to tune the control
#' @param SQBalgorithm.1 Use for the initial training. Option: lm, CART(default), knnreg, nnet, PCR.
#' @param SQBalgorithm.2 Use for the last training. Option: lm, CART(default), knnreg, nnet, PCR.
#' @param k The number of nearest neighbour used for knnreg
#' @param ncomp The number of component used for PCR
#' @param nnet.size The number of hidden layer and neuron for nnet
#' @importFrom rpart rpart
#' @importFrom rpart rpart.control
#' @importFrom parallel mclapply
#' @import stats
#' @import utils
#' @importFrom caret knnreg
#' @importFrom nnet nnet
#' @importFrom pls pcr
#' @return Given testing set input, make a regression prediction
#' @references Breiman L., Friedman J. H., Olshen R. A., and Stone, C. J. (1984)
#' \emph{Classification and Regression Trees.}

#' @references Soleymani, M. and Lee S.M.S(2014). Sequential combination of weighted and nonparametric bagging for classification. \emph{Biometrika}, 101, 2, pp. 491--498.
#' @references Efron, B. (1979). Bootstrap methods: Another lo ok at the jackknife. Ann. Statist., 7(1):1-26.
#' @examples
#' data(hills, package="MASS")
#' rt.df <- hills[sample(nrow(hills)),]
#' data.train <- rt.df[1 : (length(rt.df[, 1]) - 1), ]
#' data.test <- rt.df[-(1 : (length(rt.df[, 1]) - 1)),]

#' fit <- SQBreg(data.train, data.test, reps = 30, y = "time")
#' fit
#'
#' @export



SQBreg <- function(data.train, data.test, y, res, reps,
                   cores, FunKDE, control, SQBalgorithm.1, SQBalgorithm.2, k, ncomp, nnet.size){

  pb <- txtProgressBar(min = 0, max = 4, style = 3)  #

  data.train = data.train[,c(which(colnames(data.train) != y), which(colnames(data.train) == y))]
  data.test = data.test[,c(which(colnames(data.test) != y), which(colnames(data.test)==y))]

  mingzi <- paste("x", 1 : length(data.train), sep = "")
  mingzi[length(mingzi)] <- "y1"
  colnames(data.train) <- mingzi
  colnames(data.test) <- mingzi
  formula <- y1 ~ .
  n.predictors = ncol(data.train) - 1
  maxit <- nrow(data.train)  #  input data size#
  index <- 1 : maxit
  response <- n.predictors + 1  #  The column position of response

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
    control = list(minsplit = 10, cp=0) # CART
  }

  if (missing(k)){
    k = 2 # KNN
  }

  if (missing(nnet.size) || nnet.size < 2){
    warning('Invalid hidden layer or neuron, system modify to 2')
    nnet.size = 2
  }

  if (missing(ncomp) || ncomp == 0 || n.predictors == 1){
      ncomp = 1
    }
    if (ncomp == n.predictors && n.predictors != 1){
      ncomp = n.predictors - 1
    }


  if (missing(res)){
    res = round(maxit / 2)
  }

  if (maxit <= res - 3){
    warning('Function modifies the number of resampling that it is too close or over the training size.')
    res = maxit - 4
  }

  if (missing(reps)){
    reps = 100
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
        fit1.step2.lm1 <- knnreg(formula = formula, data = bootstrap.sample1, k = k)
      }

      if (SQBalgorithm == "nnet"){
        fit1.step2.lm1 <- nnet(formula = formula, size = nnet.size, data = bootstrap.sample1, linout = T, trace = F)
      }

      if (SQBalgorithm == "PCR"){
        fit1.step2.lm1 <- pcr(formula, data = bootstrap.sample1, scale =TRUE, validation = "CV")
      }

      pred.lm <- as.numeric(predict(fit1.step2.lm1, newdata=data.train[j, ], ncomp = ncomp))
      store[i] <- pred.lm
    }
    store
  }


  #   1 cutoff
  setTxtProgressBar(pb, 1)

  if (missing(cores) || cores == 1){
    cores = F
    res.replicate <- replicate(reps, RegTree(formula=formula, data.train, res=res, index = index,
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
      RegTree(formula=formula, data.train, res, index=index,
              SQBalgorithm = SQBalgorithm.1)},
      mc.cores = cores)
    res.replicate <- matrix(unlist(res.replicate), ncol = reps)

  }

  else if (cores > getOption("mc.cores", parallel::detectCores()) || cores < 1 || cores %% 1 != 0){
    stop("The use number of cores is invalid")
  }

  setTxtProgressBar(pb, 2)

  new.reg.lm100 <- res.replicate

  setTxtProgressBar(pb, 3)
  if (missing(FunKDE) || FunKDE == "logistic")
  {
    FunKDE = function(new.reg.lm100, reps, SIGMA) {
      if (missing(SIGMA)){
        SIGMA <- sqrt(pi ^ 2 / 3) #
      }
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
    #
    FunKDE = function(new.reg.lm100, reps, SIGMA) {
      if (missing(SIGMA)){
        SIGMA <- 1 #
      }

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
  #
  else if (FunKDE == "rectangle")
  {
    FunKDE = function(new.reg.lm100, reps, SIGMA) {
      if (missing(SIGMA)){
        SIGMA <- sqrt(1/3) #
      }
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

  else {
    stop("response generator invalid")
  }
  #
  KDE100.generator150 <- FunKDE(new.reg.lm100, reps)
  KDEtraining.newL <- data.frame(data.train[, -length(data.train), drop = FALSE], KDE100.generator150)
  KDE.fit <- matrix(0, nrow = nrow(data.test) + nrow(data.train) - maxit, ncol = reps)

  if(SQBalgorithm.2 == "CART"){
    for (i in (n.predictors + 1) : (reps + n.predictors)) {
      gongshi.KDE <- as.formula(paste("KDEtraining.newL[, i] ~",
                                      paste(attr(terms.formula(formula, data = data.train), "term.labels"),
                                            sep = "", collapse = "+")))
      KDE.model <- rpart(formula = gongshi.KDE, data = KDEtraining.newL, method="anova")
      KDE.fit[, i - n.predictors] <- predict(KDE.model, newdata=data.test)
    }
  }

  if(SQBalgorithm.2 == "lm"){
    for (i in (n.predictors + 1) : (reps + n.predictors)) {
      gongshi.KDE <- as.formula(paste("KDEtraining.newL[, i] ~",
                                      paste(attr(terms.formula(formula, data = data.train), "term.labels"),
                                            sep = "", collapse = "+")))
      KDE.model <- lm(formula = gongshi.KDE, data = KDEtraining.newL)
      KDE.fit[, i - n.predictors] <- predict(KDE.model, newdata=data.test)
    }
  }

  if(SQBalgorithm.2 == "KNN"){
    for (i in (n.predictors + 1) : (reps + n.predictors)) {
      gongshi.KDE <- as.formula(paste("KDEtraining.newL[, i] ~",
                                      paste(attr(terms.formula(formula, data = data.train), "term.labels"),
                                            sep = "", collapse = "+")))
      KDE.model <- knnreg(formula = gongshi.KDE, data = KDEtraining.newL, k = k)
      KDE.fit[, i - n.predictors] <- predict(KDE.model, newdata=data.test)
    }
  }

  if(SQBalgorithm.2 == "nnet"){
    for (i in (n.predictors + 1) : (reps + n.predictors)) {
      gongshi.KDE <- as.formula(paste("KDEtraining.newL[, i] ~",
                                      paste(attr(terms.formula(formula, data = data.train), "term.labels"),
                                            sep = "", collapse = "+")))
      KDE.model <- nnet(formula = gongshi.KDE, data = KDEtraining.newL, size = nnet.size, linout = T, trace = F)
      KDE.fit[, i - n.predictors] <- as.numeric(predict(KDE.model, newdata=data.test))
      # #
    }
  }

  if(SQBalgorithm.2 == "PCR"){
    for (i in (n.predictors + 1) : (reps + n.predictors)) {
      gongshi.KDE <- as.formula(paste("KDEtraining.newL[, i] ~",
                                      paste(attr(terms.formula(formula, data = data.train), "term.labels"),
                                            sep = "", collapse = "+")))
      KDE.model <- pcr(formula = gongshi.KDE, data = KDEtraining.newL, scale =TRUE, validation = "CV")
      KDE.fit[, i - n.predictors] <- as.numeric(predict(KDE.model, newdata=data.test, ncomp = ncomp))
      # #
    }
  }
  #   3 cutoff

  KDE.fit <- as.data.frame(KDE.fit)
  final.prediction <- rowMeans(KDE.fit) #####################  Prediction results for testset.

  #   4 cutoff
  setTxtProgressBar(pb, 4)
  return(final.prediction)
}


