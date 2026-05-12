# ==============================================================================
# BASIC & STATS
# ==============================================================================

## str_removeM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#iteratively removes matchin strings
str_removeM <- function(s,s_r){s <- s[!s %in% s_r];return(s)}

## str_get~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# detect and extract at the sametime
str_get <- function(s,p){s[stringr::str_detect(s,p)]}


## listN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#make a list and assign a name

## ll~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# display list content nicely
lkL <- function (l){Hmisc::list.tree(l)}

## is.nanM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#apply is.nan to all columns
is.nanM<-function(x)do.call(cbind, lapply(x, is.nan))

## deblank~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#removes all empty strings
deblank<-function(x){x[x!=""]}

## list.unstack~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Restructure list of list (often the outputs from foreach mcapply) by names
# check other useful functions in rlist package
list.unstack<-function(r) {as.list(as.data.frame(do.call(rbind, r)))}

## list.rename~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rename specific element in list
list.rename<-function(r,n_old,n_new) {
  browser()
  
  as.list(as.data.frame(do.call(rbind, r)))
}

## dimAdj~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Adjust dimension of matrixes
# I = (d=n dimentional matrix,dc=dimensions to collapse) 
dimAdj<-function(d,dc){
  dims<-dim(d);
  ndim<-length(dims);
  dp<-aperm(d,c((1:ndim)[-dc],dc))#force requested dims to be last
  dim(dp)<-c(dims[-dc],prod(dims[dc]))
  return(dp)
}

## idxDR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# index function for nd array/matrix by an indexer(logical) and a dimension to apply
# this works but it is old and it uses "eval, use %idxDR% if possible
idxDR <- function(d,idxL,dim2act){
  # basic settings
  ndim<-length(dim(d))
  a <- rep("",1,ndim);
  b<-paste(rep(",",1,ndim))
  if (length(idxL)!=length(dim2act)){stop("list of index must match # of dimensions to act!")}
  
  # Prepare parser
  for (dd in 1:length(dim2act)){a[dim2act[dd]]<-paste0("idxL",sprintf("[[%d]]",dd))}
  dimF <- paste(head(c(rbind(a,b)),-1),collapse="")
  parser <- paste("d[",dimF,"]",collapse = "")
  
  # Index
  d_p <- eval(parse(text=parser))
  return(d_p)
}


# # OLD: not sensitive to multiple index applied to multiple dimensios
# idxDR <- function(d,idx,dim2act){
#   dimF <- paste(rep(",",1,length(dim(d))-1))
#   dimF <- paste(append(dimF,"idx",dim2act-1),collapse = "")
#   parser <- paste("d[",dimF,"]",collapse = "")
#   d_p <- eval(parse(text=parser))
#   return(d_p)
# }

## findCI~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gives back indexes of closest value in template(shows warning messages if there are multiple candidates...)
findCI <- function(v, tmp){
  r<-vector(length=length(v))
  for (i in 1:length(v)){r[i]<-which(abs(tmp-v[i])==min(abs(tmp-v[i])))}
  return(r)
}

# dimc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# one linear of dim changes dimension of input into a specified shape
# -x = a vector, matrix or array
# -d = dimensions of the output
dimc <- function(x,d){dim(x) <- d;return(x)}

# nearest~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nearest<-function(v,tmp){
  r<-vector(length=length(v))
  for (i in 1:length(v)){v[i]<-which.min(abs(tmp-v[i]))};
  return(v)
}

# %circshift: shift vector a at sz position-------------------------------------
circshift <- function(a, sz) {
  if (is.null(a)) return(a)
  
  if (is.vector(a) && length(sz) == 1) {
    n <- length(a)
    s <- sz %% n
    a <- a[(1:n-s-1) %% n + 1]
    
  } else if (is.matrix(a) && length(sz) == 2) {
    n <- nrow(a); m <- ncol(a)
    s1 <- sz[1] %% n
    s2 <- sz[2] %% m
    a <- a[(1:n-s1-1) %% n + 1, (1:m-s2-1) %% m + 1]
  } else
    stop("Length of 'sz' must be equal to the no. of dimensions of 'a'.")
  
  return(a)
}

# %wrapToPi: Wrap angle in radians to [-pi pi]----------------------------------
# %   lambdaWrapped = wrapToPi(LAMBDA) wraps angles in LAMBDA, in radians,
# %   to the interval [-pi pi] such that pi maps to pi and -pi maps to
# %   -pi.  (In general, odd, positive multiples of pi map to pi and odd,
#            %   negative multiples of pi map to -pi.)
# %
# %   See also wrapTo2Pi, wrapTo180, wrapTo360.
wrapTo2Pi <- function(x){
  i <- x > 0
  x <- x %% (2*pi)
  x[(x == 0) & i] <- 2*pi
  return(x)
}

# %wrapToPi: Wrap angle in radians to [-pi pi]----------------------------------
# %   lambdaWrapped = wrapToPi(LAMBDA) wraps angles in LAMBDA, in radians,
# %   to the interval [-pi pi] such that pi maps to pi and -pi maps to
# %   -pi.  (In general, odd, positive multiples of pi map to pi and odd,
#            %   negative multiples of pi map to -pi.)
# %
# %   See also wrapTo2Pi, wrapTo180, wrapTo360.
wrapToPi <- function(x){
  i <- (x < -pi) | (pi < x)
  x[i] <- wrapTo2Pi(x[i] + pi) - pi;
  return(x)
}

# %wrapTo360: Wrap angle in degrees to [0 360]----------------------------------
# %
# %   lonWrapped = wrapTo360(LON) wraps angles in LON, in degrees, to the
# %   interval [0 360] such that zero maps to zero and 360 maps to 360.
# %   (In general, positive multiples of 360 map to 360 and negative
#      %   multiples of 360 map to zero.)
# %
# %   See also wrapTo180, wrapToPi, wrapTo2Pi.
wrapTo360 <- function(x){
  i <- x > 0
  x <- x %% 360
  x[x==0 & i] <- 360
  return(x)
}

# %wrapTo180: Wrap angle in degrees to [-180 180]-------------------------------
# %
# %   lonWrapped = wrapTo180(LON) wraps angles in LON, in degrees, to the
# %   interval [-180 180] such that 180 maps to 180 and -180 maps to -180.
# %   (In general, odd, positive multiples of 180 map to 180 and odd,
#      %   negative multiples of 180 map to -180.)
# %
# %   See also wrapTo360, wrapTo2Pi, wrapToPi.
wrapTo180 <- function(x){
  i <- (x -180) | (180 < x)
  x[i] = wrapTo360(x[i] + 180) - 180
  return(x)
}

# gapfill~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# interleave character vector with another character (e.g., "a" "b" -> "a,b")
# -s = a character vector 
# -f = a filler
# EX) gapfill(c("a","b","c"),"@")
gapfill <- function(s,f){do.call(paste, c(as.list(s), sep = f))}

## findSegments~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# detect segments based on threshold 
findSegments <- function(x, threshold){
  hit <- which(x > threshold)
  n <- length(hit)
  ind <- which(hit[-1] - hit[-n] > 1)
  starts <- c(hit[1], hit[ ind+1 ])
  ends <- c(hit[ ind ], hit[n])
  cbind(starts,ends)
}

## means.along~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply averaging to the specific dimension of matrix

means.along <- function(a, i) {
  n <- length(dim(a))
  b <- aperm(a, c(seq_len(n)[-i], i))
  rowMeans(b, dims = n - 1, na.rm=T)
}

## lagfunc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lagfunc <-function(var,numpositions)
{
  TEMP <-c()
  for(i in 1:numpositions) TEMP<-append(TEMP,NA)
  TEMP <- c(TEMP,var)
  TEMP<-TEMP[1:(length(TEMP)-numpositions)] 
  return(TEMP)
}

## apermWW~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#apermWW applies aperm but specicied by strings
apermWW <-function(d,oldD,newD)
{
  swpD<-unlist(lapply(as.list(newD), function(s){grep(s,oldD)}))
  d<-aperm(d,swpD)
  return(d)
}

## rtnorm~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -n = number of data to draw
# -mean = mean of normal distribution
# -sd = standard deviation
# -a = lower bound
# -b = upper bound
# Truncated normal distribution
# rtnorm <- function(n, m, s, lwr, upr, rounding) {
#   samp <- round(rnorm(n, m, s), rounding)
#   samp[samp < lwr] <- lwr
#   samp[samp > upr] <- upr
#   samp
# }

rtnorm <- function(n, mean, sd, a = -Inf, b = Inf){
  qnorm(runif(n, pnorm(a, mean, sd), pnorm(b, mean, sd)), mean, sd)
}

## effect size for efex~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
effS<-function(r){r$ANOVA$MSE = r$ANOVA$SSd/r$ANOVA$DFd;r$ANOVA$eta <- r$ANOVA$SSn/(r$ANOVA$SSn+r$ANOVA$SSd);return(r)};


## norm_for_wse~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#function for normalizing for within-subject error bars
norm_for_wse <- function(dv,ID){
  dat <- tibble(dv = dv,ID=ID)
  dat2 <- dat %>%
    mutate(grand_mean = mean(dv,na.rm=T)) %>%
    group_by(ID) %>%
    mutate(submean = mean(dv,na.rm=T)) %>%
    ungroup() %>%
    mutate(normdv = dv - submean + grand_mean)
  return(dat2$normdv)
}

## summarySE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Gives count, mean, standard deviation, standard error of the mean, and confidence interval (default 95%).
##   data: a data frame.
##   measurevar: the name of a column that contains the variable to be summariezed
##   groupvars: a vector containing names of columns that contain grouping variables
##   na.rm: a boolean that indicates whether to ignore NA's
##   conf.interval: the percent range of the confidence interval (default is 95%)
summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  require(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(N    = length2(xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     sd   = sd     (xx[[col]], na.rm=na.rm)
                   )
                 },
                 measurevar
  )
  
  # Rename the "mean" column    
  datac <- rename(datac, c("mean" = measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}

## Summarizes data, handling within-subjects variables by removing inter-subject variability.
## It will still work if there are no within-S variables.
## Gives count, un-normed mean, normed mean (with same between-group mean),
##   standard deviation, standard error of the mean, and confidence interval.
## If there are within-subject variables, calculate adjusted values using method from Morey (2008).
##   data: a data frame.
##   measurevar: the name of a column that contains the variable to be summariezed
##   betweenvars: a vector containing names of columns that are between-subjects variables
##   withinvars: a vector containing names of columns that are within-subjects variables
##   idvar: the name of a column that identifies each subject (or matched subjects)
##   na.rm: a boolean that indicates whether to ignore NA's
##   conf.interval: the percent range of the confidence interval (default is 95%)
summarySEwithin <- function(data=NULL, measurevar, betweenvars=NULL, withinvars=NULL,
                            idvar=NULL, na.rm=FALSE, conf.interval=.95, .drop=TRUE) {
  
  # Ensure that the betweenvars and withinvars are factors
  factorvars <- vapply(data[, c(betweenvars, withinvars), drop=FALSE],
                       FUN=is.factor, FUN.VALUE=logical(1))
  
  if (!all(factorvars)) {
    nonfactorvars <- names(factorvars)[!factorvars]
    message("Automatically converting the following non-factors to factors: ",
            paste(nonfactorvars, collapse = ", "))
    data[nonfactorvars] <- lapply(data[nonfactorvars], factor)
  }
  
  # Get the means from the un-normed data
  datac <- summarySE(data, measurevar, groupvars=c(betweenvars, withinvars),
                     na.rm=na.rm, conf.interval=conf.interval, .drop=.drop)
  
  # Drop all the unused columns (these will be calculated with normed data)
  datac$sd <- NULL
  datac$se <- NULL
  datac$ci <- NULL
  
  # Norm each subject's data
  ndata <- normDataWithin(data, idvar, measurevar, betweenvars, na.rm, .drop=.drop)
  
  # This is the name of the new column
  measurevar_n <- paste(measurevar, "_norm", sep="")
  
  # Collapse the normed data - now we can treat between and within vars the same
  ndatac <- summarySE(ndata, measurevar_n, groupvars=c(betweenvars, withinvars),
                      na.rm=na.rm, conf.interval=conf.interval, .drop=.drop)
  
  # Apply correction from Morey (2008) to the standard error and confidence interval
  #  Get the product of the number of conditions of within-S variables
  nWithinGroups    <- prod(vapply(ndatac[,withinvars, drop=FALSE], FUN=nlevels,
                                  FUN.VALUE=numeric(1)))
  correctionFactor <- sqrt( nWithinGroups / (nWithinGroups-1) )
  
  # Apply the correction factor
  ndatac$sd <- ndatac$sd * correctionFactor
  ndatac$se <- ndatac$se * correctionFactor
  ndatac$ci <- ndatac$ci * correctionFactor
  
  # Combine the un-normed means with the normed results
  merge(datac, ndatac)
}

## normWS & normDS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get normalized value for within-subject error (watch out for 3 levels!)
normWS<-function(d,s,dv){eval(substitute(d%>%group_by(s)%>%mutate(sAV=mean(dv,na.rm=TRUE))%>%
                                           ungroup()%>%mutate(gAV=mean(dv,na.rm=TRUE))%>%
                                           mutate(DV_n=dv-sAV+gAV),
                                         list(s=as.name(s),dv=as.name(dv)))) %>% data.table()}

normDS<-function(d,s,dv){eval(substitute(d%>%group_by(s)%>%summarise(sAV=mean(dv,na.rm=TRUE))%>%
                                           mutate(gAV=mean(sAV,na.rm=TRUE)) %>%
                                           mutate(DV_n=dv-sAV+gAV),
                                         list(s=as.name(s),dv=as.name(dv))))}

# the old version of norm function : using sAV to get gAV....
# normWS_old<-function(d,s,dv){eval(substitute(d%>%group_by(s)%>%mutate(sAV=mean(dv))%>%
#                                            mutate(gAV=mean(sAV))%>%
#                                            mutate(DV_n=dv-sAV+gAV),
#                                           list(s=as.name(s),dv=as.name(dv))))}
# 
# Example) # # Switch * Music position (new!)
# rt_pos<-subset(ds,MCONTEXT=="Music")%>%group_by(SUBID,SWITCH)%>%summarise(DV=mean(RT,na.rm=TRUE))%>%
#   normWS("SUBID","DV")%>%group_by(SWITCH)%>%summarize(DVm=mean(DV),se=sd(DV_n)/sqrt(n()),wseb=se*1.96,n=n());print(rt_pos)

# Test data (http://www.cogsci.nl/blog/tutorials/156-an-easy-way-to-create-graphs-with-within-subject-error-bars)
# ds_test<-data.frame(SUBID=rep(1:6,2),value=c(2,1,8,3,7,3,5,2,9,6,9,5),c=rep(c("B","A"),each=6))
# ds_test%>%normWS("SUBID","value")%>%group_by(c)%>%summarize(DVm=mean(DV_n),se=sd(DV_n)/sqrt(n()),wseb=se*1.96,n=n())

## deviance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#deviance test for multilevel model comparisons 

deviance <- function(a, b) {
  diffneg2LL <- (-2*as.numeric(logLik(a))) - (-2*as.numeric(logLik(b)))
  dfneg2LL <- (attr(logLik(b), "df") - attr(logLik(a), "df"))
  p<-(1 - pchisq(diffneg2LL, dfneg2LL))
  return(print(paste("The -2LL difference is ", round(diffneg2LL, digits=3), "with ", dfneg2LL, "df, p = ", round(p, digits=3))))
}

## std_beta_MLM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract case-specific coefficients + standardized coefs in tidy format for MLM models
# Currently, it only works for level2. Not pipable!! EX) r<-std_beta_MLM(m)
std_beta_MLM <-function(fit, ci.lvl = .95,type = "std", rmITC = T,..){
  # Standardized coefficients + ci
  # code from Ben Bolker + sjstats
  # http://stackoverflow.com/a/26206119/2094622
  # https://github.com/strengejacke/sjstats/blob/master/R/std_b.R
  # arm::standardize()->https://cran.r-project.org/web/packages/arm/arm.pdf
  sdy <- stats::sd(lme4::getME(fit, "y"))
  sdx <- apply(lme4::getME(fit, "X"), 2, sd)
  # sc <- lme4::fixef(fit) * sdx / sdy
  # se.fixef <- stats::coef(summary(fit))[, "Std. Error"]
  # se <- se.fixef * sdx / sdy
  #r1 <- std_beta(fit,ci.lvl=ci.lvl,type=type);#older than R3.5
  r1 <- effectsize::standardize_parameters(fit,ci.lvl=ci.lvl,type=type,method="refit") # basic,refit
  r1 <- data.table(r1,SE = attr(r1, "standard_error"))
  colnames(r1)<-c("term","coef_std","CI","CI_low","CI_high","SE")
  
  # Keep statistic value (tvalue or zvalue)
  s <- broom.mixed::tidy(fit,effects="fixed") %>% data.table()
  r1$statistic <- s$statistic[s$term %in% r1$term]
  r1$coef <- s$estimate;# original coefficient
  
  # Standardized coeffs for each case 
  c <-coef(fit);cn=names(c)
  feL <- setNames(split(c[[cn]], seq(nrow(c[[cn]]))), rownames(c[[cn]]))
  sc_i<-dplyr::bind_rows(lapply(feL,function(f){f * sdx/sdy}),.id = cn)#sc for all cases
  r2=tidyr::gather(sc_i,term,std.estimate,-1)
  if (rmITC){r2<-r2 %>% dplyr::filter(term!="(Intercept)")}
  
  # Output
  std.coef <- setClass("std.coef", slots = c(fixef="data.frame", coef="data.frame"))
  return(new("std.coef",fixef=r1,coef=r2))
}

## std_beta_MLM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract case-specific coefficients + standardized coefs in tidy format for MLM models
# Currently, it only works for level2. Not pipable!! EX) r<-std_beta_MLM(m)
std_beta_MLM2 <-  function(fit, ci.lvl = .95,type = "std", rmITC = T,..){
  # Standardized coefficients + ci
  # code from Ben Bolker + sjstats
  # http://stackoverflow.com/a/26206119/2094622
  # https://github.com/strengejacke/sjstats/blob/master/R/std_b.R
  # arm::standardize()->https://cran.r-project.org/web/packages/arm/arm.pdf
  sdy <- stats::sd(lme4::getME(fit, "y"))
  sdx <- apply(lme4::getME(fit, "X"), 2, sd)
  # sc <- lme4::fixef(fit) * sdx / sdy
  # se.fixef <- stats::coef(summary(fit))[, "Std. Error"]
  # se <- se.fixef * sdx / sdy
  #r1 <- std_beta(fit,ci.lvl=ci.lvl,type=type);#older than R3.5
  r1 <- effectsize::standardize_parameters(fit,ci.lvl=ci.lvl,type=type,method="refit") # basic,refit
  r1 <- data.table(r1,SE = attr(r1, "standard_error"))
  colnames(r1)<-c("term","coef_std","CI","CI_low","CI_high","SE")
  
  # Keep statistic value (tvalue or zvalue)
  s <- broom.mixed::tidy(fit,effects="fixed") %>% data.table()
  r1$statistic <- s$statistic[s$term %in% r1$term]
  r1$coef <- s$estimate;# original coefficient
  
  # Standardized coeffs for each case 
  c <-coef(fit);cn=names(c);r2L <- list()
  for (lv in cn){
    feL <- setNames(split(c[[lv]], seq(nrow(c[[lv]]))), rownames(c[[lv]]))        
    sc_i<-dplyr::bind_rows(lapply(feL,function(f){f * sdx/sdy}),.id = lv)#sc for all cases
    r2=tidyr::gather(sc_i,term,std.estimate,-1)
    if (rmITC){r2<-r2 %>% dplyr::filter(term!="(Intercept)")}
    r2L[lv] <- list(r2) # create a list of coefficients for nexted levels
  }
  
  # Output
  std.coef <- setClass("std.coef", slots = c(fixef="data.frame", coefL="list"))
  return(new("std.coef",fixef=r1, coefL=r2L))
}


## "%=check%" parsing list output to multiple variables ~~~~~~~~~~~~~~~~~~~~~~~~
# Assigns value to the variable if that variable does not exist (built-in if condition)
"%=check%" <- function(x, y) {
  Var <- deparse(substitute(x))
  if (!exists(Var)) {assign(Var, y, parent.frame())}
}

## %=% parsing list output to multiple variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %-%, extendToMatch(), g()
# EX) g(a, b, c)  %=%  list("hello", 123, list("apples, oranges"))

# Generic form
'%=%' = function(l, r, ...) UseMethod('%=%')

# Binary Operator
'%=%.lbunch' = function(l, r, ...) {
  Envir = as.environment(-1)
  
  if (length(r) > length(l))
    warning("RHS has more args than LHS. Only first", length(l), "used.")
  
  if (length(l) > length(r))  {
    warning("LHS has more args than RHS. RHS will be repeated.")
    r <- extendToMatch(r, l)
  }
  
  for (II in 1:length(l)) {
    do.call('<-', list(l[[II]], r[[II]]), envir=Envir)
  }
}

# Used if LHS is larger than RHS
extendToMatch <- function(source, destin) {
  s <- length(source)
  d <- length(destin)
  
  # Assume that destin is a length when it is a single number and source is not
  if(d==1 && s>1 && !is.null(as.numeric(destin)))
    d <- destin
  
  dif <- d - s
  if (dif > 0) {
    source <- rep(source, ceiling(d/s))[1:d]
  }
  return (source)
}

# Grouping the left hand side
g = function(...) {
  List = as.list(substitute(list(...)))[-1L]
  class(List) = 'lbunch'
  return(List)
}


## first/last ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# index first/last n element in the list 

first <- function(x, n=1, ...) head(x, n=n, ...)
last  <- function(x, n=1, ...) tail(x, n=n, ...)

"first<-" <- function(x, n=1, ..., value )
{
  x[1:n] <- value[1:n]
  x
}

"last<-" <- function(x, n=1, ..., value )
{
  index <- seq( length(x)-n+1, length(x) )
  x[index] <- value[1:n]
  x
}

## ll ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# simillar to "whos" function in Matlab (taken from gdata package)
ll <- function(pos=1, unit="KB", digits=0, dim=FALSE, sort=FALSE, class=NULL,
               invert=FALSE, ...)
{
  get.object.class <- function(object.name, pos)
  {
    object <- get(object.name, pos=pos)
    class <- class(object)[1]
    return(class)
  }
  
  get.object.dim <- function(object.name, pos)
  {
    object <- get(object.name, pos=pos)
    if(class(object)[1] == "function")
      dim <- ""
    else if(!is.null(dim(object)))
      dim <- paste(dim(object), collapse=" x ")
    else
      dim <- length(object)
    return(dim)
  }
  
  get.object.size <- function(object.name, pos)
  {
    object <- get(object.name, pos=pos)
    size <- try(unclass(utils::object.size(object)), silent=TRUE)
    if(class(size) == "try-error")
      size <- 0
    return(size)
  }
  
  ## 1  Set unit, denominator, original.rank
  unit <- match.arg(unit, c("bytes","KB","MB"))
  denominator <- switch(unit, "KB"=1024, "MB"=1024^2, 1)
  original.rank <- NULL
  
  ## 2  Detect what 'pos' is like, then get class, size, dim
  if(is.character(pos))  # pos is an environment name
    pos <- match(pos, search())
  if(is.list(pos))  # pos is a list-like object
  {
    if(is.null(names(pos)))
      stop("All elements of a list must be named")
    original.rank <- rank(names(pos))
    pos <- as.environment(pos)
  }
  if(length(ls(pos,...)) == 0)  # pos is an empty environment
  {
    object.frame <- data.frame()
  }
  else if(environmentName(as.environment(pos)) == "Autoloads")
  {
    object.frame <- data.frame(rep("function",length(ls(pos,...))),
                               rep(0,length(ls(pos,...))),
                               row.names=ls(pos,...))
    if(dim)
    {
      object.frame <- cbind(object.frame, rep("",nrow(object.frame)))
      names(object.frame) <- c("Class", unit, "Dim")
    }
    else
      names(object.frame) <- c("Class", unit)
  }
  else
  {
    class.vector <- sapply(ls(pos,...), get.object.class, pos=pos)
    size.vector <- sapply(ls(pos,...), get.object.size, pos=pos)
    size.vector <- round(size.vector/denominator, digits)
    object.frame <- data.frame(class.vector=class.vector,
                               size.vector=size.vector,
                               row.names=names(size.vector))
    names(object.frame) <- c("Class", unit)
    if(dim)
      object.frame <- cbind(object.frame,
                            Dim=sapply(ls(pos,...),get.object.dim,pos=pos))
  }
  
  ## 3  Retain original order of list elements
  if(!sort && !is.null(original.rank))
    object.frame <- object.frame[original.rank,]
  
  ## 4  Filter results given class
  if(!is.null(class))
  {
    include <- object.frame$Class %in% class
    if(invert)
      include <- !include
    object.frame <- object.frame[include,]
  }
  
  return(object.frame)
}

## %idxDR% flexibly index any dimensions of arrays ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Ex) 
# x <- array(1:30, c(2,3,5))
# print(x)
# x %idxDR% list(TRUE,1,2) # is same as writing x[,1,2]
`%idxDR%` <- function(x,idx) {
  do.call('[', c(list(x), idx))
}


# ==============================================================================
# DECODING & ENCODING MODELS
# ==============================================================================

# LETTERS_ex:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# adjuted LETTER function compatible for any number of levels
LETTERS_ex <- function(n) {
  unlist(Reduce(paste0, replicate(n %/% length(LETTERS), 
                                  LETTERS, simplify=FALSE),
                init=LETTERS,accumulate=TRUE))[1:n] 
}

# best_(whatever)---------------------------------------------------------------
# lazy way to exctract best tuned parameters
best_pda<-function(m){m$results$lambda==as.character(m$bestTune)};


# preprop_decode----------------------------------------------------------------
# applies 1) filtering, 2) cutting, 3) balancing(ignores fold size), and 4) factorization
# 2) - 3) are optional, step 1) and 4) are always performed. 
# ds = template ds (e.g., beh data) to be merged to neural data
# dsN = neural data (eeg) assuming [trial,time,freq,elec] <- fix this eventually
# pset = setting for preprocessing 
preprop_decode <- function(ds,dsN,pset) {
  # Setting
  set.seed(612)
  modelV = pset$modelV;
  filtIDX = pset$filtIDX;
  cutV = pset$cutV;
  balanceV = pset$balanceV;
  crossV = pset$crossV;
  if (length(crossV) == 0 || all(is.na(crossV))) crossV <- NULL
  foldsL = list();
  
  # Filtering: filter out observations condition
  #Trial,Time,Frequency,Electrode or Trial,Time,Electrode: still assume 1=trial
  ds[, filtIDX := filtIDX]
  if (!is.null(dsN)){dsN<-idxDR(dsN,list(filtIDX),1);}else{dsN=filtIDX}
  ds<-ds[filtIDX==T]
  ds[,rowIndex:=.I]
  
  #Cutting: adjust # of observations at a fixed value
  if (!is.na(cutV)){
    if (dim(ds)[1] < cutV) {cutV = dim(ds)[1]}
    ind<-sample(1:dim(ds)[1],cutV)
    ds<-slice(ds,ind);
    if (!is.null(dsN)){dsN<-dsN[ind,,,]}
    print(paste0("Data set is cut off to be ", cutV));
  }
  
  #Factorize to-be-modeled variable (and use arbitrary characters for each level)
  for (v in modelV){
    ALPHABETS <- LETTERS_ex(length(unique(ds[[v]])))  
    if (is.character(ds[[v]])) ds[[v]] <- as.numeric(as.factor(ds[[v]]))
    if (min(ds[[v]]) == 0) ds[[v]] <- ds[[v]] + 1
    ds[[v]] <- ALPHABETS[as.matrix(ds[[v]])]
    ds[[v]] <- factor(unlist(ds[[v]]), levels = ALPHABETS)
    ds[[v]]<-droplevels(ds[[v]])
  }
  
  # (Old)Balancing: equalize # of trials levels of a set of variables
  #if (any(!is.na(balanceV))){
  # #Prepare index to merge back to the original data after sub-sampling with lowest # of data
  # set.seed(612);#Get same sets of trials for each subject
  # ds[,`:=`(balanceID=.GRP, balanceN=.N,obsID=.I),by=balanceV]
  # ind<-ds[,.SD[sample(.N, min(min(ds$balanceN),.N))],by=balanceID]
  # setorderv(ind,c(colnames(ds),"obsID"))
  # if (!is.null(dsN)){dsN<-dsN[ind$obsID,,,]};#Trial,Time,Frequency,Electrode
  # ds<-ind[,-c("balanceID","balanceN","obsID")];#CAUTION!! ALWAYS FILTER dsN first!!!
  # print(paste0("Data set is balanced by ", paste0(balanceV,collapse="_")," variable...!!"));
  #}
  
  # ---- crossV が指定されていて balanceV が未指定なら、balanceV = crossV ----
  if (!is.null(crossV) && !any(!is.na(balanceV))) { balanceV <- crossV }
  
  # ---- crossV を安全化（仕様1：意味を変えない） ----
  suppressWarnings(ds[,crossV_eff:=1L])
  if  (!is.null(crossV)){ds[,crossV_eff:=get(crossV)]}

  # ---- Balancing folds ----
  if (any(!is.na(balanceV))) {
    ds[, obsID := 1:.N, by = crossV_eff]
    dsL <- split(ds, ds[["crossV_eff"]])
    k <- pset$controlL$number
    repeats <- pset$controlL$repeats
    foldsL <- list()
    
    for (g in seq_along(dsL)) {
      lev <- names(dsL)[g]
      foldsL[[lev]] <- list()

      for (r in 1:repeats) {
        dsB <- dsL[[g]][,.SD[sample(.N, min(dsL[[g]][,.N,by=balanceV]$N))],by=balanceV]
        fold_id <- dsB[, sample(rep(1:k, length.out = .N)), by = balanceV]$V1
        
        for (i in 1:k) {
          test_idx  <- dsB$obsID[fold_id == i]
          train_idx <- setdiff(dsB$obsID, test_idx)
          nm <- paste0("Fold", i, ".Rep", r)
          foldsL[[lev]][[nm]] <- train_idx
        }
      }
    }
    # Notification
    first_lev <- names(foldsL)[1]
    first_fold <- names(foldsL[[first_lev]])[1]
    n_train <- length(foldsL[[first_lev]][[first_fold]])
    crossV_label <- if (is.null(crossV)) "ALL" else crossV
    print(sprintf("Dataset is balanced by %s within %s.", balanceV, crossV_label))
    print(sprintf("Each training fold contains %d trials.",n_train))
    print(ds[, .N, by = c("crossV_eff", balanceV)])
  }
  # # OLD METHOD
  # if (any(!is.na(balanceV))) {
  #   
  #   ds[, `:=`(balanceID=.GRP, balanceN=.N, obsID=.I), by=c(balanceV, crossV_eff)]
  #   cellN <- ds[, .N, by = c(balanceV, crossV_eff)] %>% print()
  #   
  #   k <- pset$controlL$number
  #   repeats <- pset$controlL$repeats
  #   repeatsL <- paste("Rep", gsub(" ", "0", format(1:repeats)), sep = "")
  #   foldsN <- paste("Fold", gsub(" ", "0", format(1:k)), sep = "")
  #   labels <- do.call(paste0, expand.grid(foldsN, ".", repeatsL))
  #   foldsL <- vector("list", length(labels)); names(foldsL) <- labels
  #   
  #   # ★変更点：常に cross 単位で fold を作る
  #   # cross が無い場合は全体で1グループになる
  #   outer_by <- if (!is.null(crossV)) c(crossV_eff) else NULL
  #   
  #   for (i in seq_along(foldsL)) {
  #     set.seed(i)
  #     
  #     ind <- ds[
  #       , {
  #         tgt <- .SD[, .N, by = balanceV][, min(N)]
  #         .SD[, .SD[sample.int(.N, min(.N, tgt))], by = balanceV]
  #       },
  #       by = outer_by
  #     ]
  #     
  #     if (is.null(crossV)) {
  #       samples <- sort(ind$obsID)
  #     } else {
  #       samples <- lapply(split(ind$obsID, ind[, get(crossV_eff)]), sort)
  #     }
  #     
  #     foldsL[[i]] <- samples
  #   }
  #   
  #   # Notifications
  #   crossV_label <- if (is.null(crossV)) "ALL" else crossV
  #   idx <- if (is.null(crossV)) foldsL[[1]] else foldsL[[1]][[1]]
  #   
  #   print(sprintf("Dataset is balanced within each %s level across %s.", crossV_label, balanceV))
  #   print(sprintf(
  #     "Each fold contains %d trials.", length(idx)))
  # }

  # For Cross-condition decoding, split the parent data
  foldsL <- if (length(foldsL) == 1) foldsL[[1]] else foldsL
  #if (!is.null(crossV)) {ds <- split(ds, ds[[crossV]])}
  return(list(data.table(ds),dsN,list(filtIDX),foldsL)) # put filtIDX in list 
}
# preprop_decode <- function(ds,dsN,pset) {
#   # Setting
#   modelV = pset$modelV;
#   filtIDX = pset$filtIDX;
#   cutV = pset$cutV;
#   balanceV = pset$balanceV;
#   
#   # Filtering: filter out observations condition
#   #Trial,Time,Frequency,Electrode or Trial,Time,Electrode: still assume 1=trial
#   ds$filtIDX<-filtIDX;
#   if (!is.null(dsN)){dsN<-idxDR(dsN,list(filtIDX),1);}else{dsN=filtIDX}
#   ds<-ds[filtIDX==T,];
#   ds[,rowIndex:=.I]
#   
#   #Cutting: adjust # of observations at a fixed value
#   if (!is.na(cutV)){
#     if (dim(ds)[1] < cutV) {cutV = dim(ds)[1]}
#     ind<-sample(1:dim(ds)[1],cutV)
#     ds<-slice(ds,ind);
#     if (!is.null(dsN)){dsN<-dsN[ind,,,]}
#     print(paste0("Data set is cut off to be ", cutV));
#   }
#   
#   #Balancing: equalize # of trials levels of a set of variables
#   if (any(!is.na(balanceV))){
#     #Prepare index to merge back to the original data after sub-sampling with lowest # of data
#     set.seed(612);#Get same sets of trials for each subject
#     ds[,`:=`(balanceID=.GRP, balanceN=.N,obsID=.I),by=balanceV]
#     ind<-ds[,.SD[sample(.N, min(min(ds$balanceN),.N))],by=balanceID]
#     setorderv(ind,c(colnames(ds),"obsID"))
#     if (!is.null(dsN)){dsN<-dsN[ind$obsID,,,]};#Trial,Time,Frequency,Electrode
#     ds<-ind[,-c("balanceID","balanceN","obsID")];#CAUTION!! ALWAYS FILTER dsN first!!!
#     print(paste0("Data set is balanced by ", paste0(balanceV,collapse="_")," variable...!!"));
#   }
#   
#   #Factorize to-be-modeled variable (and use arbitrary characters for each level)
#   for (v in modelV){
#     ALPHABETS<-LETTERS_ex(length(c(unique(ds[[v]]))));#comes from other function
#     if (is.character(ds[[v]])){ds[[v]]<-as.numeric(as.factor(ds[[v]]))}# convert strings to number
#     if (min(ds[[v]])==0){ds[[v]]<-ds[[v]]+1} # adjust numbers starting from 0
#     ds[[v]]<-ALPHABETS[as.matrix(ds[[v]])];
#     ds[[v]]<-ds[[v]]%>%unlist%>%factor;
#     ds[[v]]<-droplevels(ds[[v]])
#   }
#   
#   return(list(ds_beh=data.table(ds),ds_eeg=dsN,filtIDX=filtIDX))
# }

# summary_decode----------------------------------------------------------------
# summarizes model outputs for 1) overall decoding accuracy, 2) posterior probability, 3) cm, 4) ROC
summary_decode <- function(m,varL,set) {
  #Default setting(using rlist package)
  set<-rlist::list.merge(list(PRED=F,CM=F,IMP=F,ROC=F,COEF=F),set)
  
  #Overall accuracy
  d<-m$trainingData # rowIndex is added within caret
  r_acc<-m$results %>% data.table()
  if (m$method=="pda"){r_acc<- m$results[best_pda(m),] %>% data.table()}
  
  #Posterior probability (be careful to use data.table... rowIndex?)
  if (set$PRED) {
    r_prob <- data.table(m$pred)
    paramName <- names(m$bestTune)[1]
    paramValue <- m$bestTune[[paramName]]
    r_prob <- r_prob[get(paramName) == paramValue]
    r_prob[, acc := (pred == obs) * 1L]
    caseC <- unique(m$levels)
    #caseC <- names(r_prob)[names(r_prob) %in% LETTERS_ex]

    # Average over CV and detect missing rows
    r_prob <- r_prob[, c("rowIndex", "acc", caseC), with = FALSE]
    r_prob <- r_prob[, lapply(.SD, mean), by = "rowIndex"][order(rowIndex)]
    miss_idx <- setdiff(seq_len(nrow(m$trainingData)), r_prob$rowIndex)
    
    # # Old method
    # r_prob<-m$pred %>% data.table(); 
    # r_prob[,acc:=(r_prob$pred==r_prob$obs)*1]
    # # keep simple accuracy 
    # caseC<-names(r_prob)[names(r_prob) %in% LETTERS] 
    # r_prob<-r_prob[,c("rowIndex","acc",caseC),with=F] 
    # r_prob<-r_prob[,lapply(.SD, mean),by="rowIndex"][order(rowIndex)] 
  }
    
  #Confusion matrix
  if (set$CM) {
    r_cm<-confusionMatrix(m$pred$pred,m$pred$obs);
    if (is.vector(r_cm$byClass)){r_cm$byClass<-t(as.matrix(r_cm$byClass))}
    r_acc$Accuracy_B <- mean(r_cm$byClass[,"Balanced Accuracy"]);# keep balanced accuracy in r_acc
    r_cm<-sweep(r_cm$table, 2, colSums(r_cm$table),FUN="/");
    r_cm<-data.table(r_cm);
  } else {r_cm=data.table()}
  
  #Feature selection & Activation map 
  if (set$IMP) {r_imp <- weight2topo(m,d,varL)}else{r_imp=data.table()}
  
  #ROC curve
  #ROC surface
  if (set$ROC) {
    r<-as.numeric(d[[".outcome"]]);
    p<-as.numeric(predict(m, d, type = 'raw'))
    r_roc<-multiclass.roc(r, p)
    r_acc$AUC<-as.numeric(r_roc$auc)
    #r_roc needs to be tidied up?(until then just a copy of r_acc)
    r_roc = r_acc;
  }else{r_roc=data.table()}
  
  #Weight(coefficient) of the best model
  if (set$COEF){
    r_coef<-coef(m$finalModel,m$bestTune) 
    r_coef<-data.table(t(r_coef)) # comparion x features
    r_coef[,COEFID:=1:.N] # Keep ID
  }else{r_coef=data.table()}
  
  #Final outputs
  r <- list(r_acc=r_acc,r_prob=r_prob,r_cm=r_cm,r_imp=r_imp,r_roc=r_roc,r_coef=r_coef)
  r <- r[sapply(r, function(x) dim(x)[1]) > 0];# remove empty slots
  return(r)
}

# summary_cross_decode for cross-condition decoding ----------------------------
summary_cross_decode <- function(m, d_c, meta_c) {
  if (nrow(d_c) == 0) return(NULL)
  p_raw  <- predict(m, newdata = d_c, type = "raw")
  p_prob <- predict(m, newdata = d_c, type = "prob")
  r_cross <- copy(meta_c)
  r_cross[, pred := p_raw]
  r_cross[, acc  := as.integer(pred == obs)]
  r_cross <- cbind(r_cross, p_prob)
  r_cross[, c("obs","pred") := NULL]
  return(list(r_prob = r_cross))
}

# summary_decodeP (using predict)-----------------------------------------------
summary_decodeP <- function(m, varL, p, ds_tst, modelV,set){
  # Initialize
  set<-rlist::list.merge(list(PRED=F,CM=F,IMP=F,ROC=F,COEF=F),set)
  
  # Index out probabilities
  if (!is.data.table(p)){p <- data.table(p)}
  idx <- as.matrix(as.numeric(ds_tst[[modelV]]))
  prob<-as.matrix(p[, names(p) %in% LETTERS,with=F])
  p[,class:=colnames(.SD)[max.col(.SD)],.SDcols=names(p)%in%LETTERS]
  p[,obs:=ds_tst[[modelV]]] # correct labels
  p[,prob:=prob[cbind(seq_along(idx),idx)]]
  p[,acc:=(class==obs)*1]
  r_prob <- ds_tst[,"rowIndex"]
  r_prob <- cbind(r_prob, p[,"acc"] ,p[,names(p)%in%LETTERS,with=F])
  
  # Get confusion matrix
  r_cm <- confusionMatrix(as.factor(p$class), as.factor(p$obs))
  Accuracy <- r_cm$overall["Accuracy"] # for later
  Accuracy_B <- mean(r_cm$byClass[,"Balanced Accuracy"]) # for later
  if (is.vector(r_cm$byClass)){r_cm$byClass<-t(as.matrix(r_cm$byClass))}
  r_cm<-data.table(sweep(r_cm$table, 2, colSums(r_cm$table),FUN="/"));
  if (!set$CM){{r_cm=data.table()}} # delete if not requested
  
  # Summarize accuracy
  r_acc <- data.table(m$bestTune)
  r_acc[["Accuracy"]] <- Accuracy
  r_acc[["Accuracy_B"]] <- Accuracy_B;
  r_acc[["Kappa"]] <- r_cm$overall["Kappa"]
  r_acc[["AccuracySD"]] <- NA # requires resampling
  r_acc[["KappaSD"]] <- NA # requires resampling
  
  #Feature selection & Activation map 
  if (set$IMP) {r_imp <- weight2topo(m,d,varL)}else{r_imp=data.table()}
  
  # ROC CURVE: WIP...
  r_roc=data.table()
  
  
  #Weight(coefficient) of the best model
  if (set$COEF){
    r_coef<-coef(m$finalModel,m$bestTune) 
    r_coef<-data.table(t(r_coef)) # comparion x features
    r_coef[,COEFID:=1:.N] # Keep ID
  }else{r_coef=data.table()}
  
  # Final ouputs
  r <- list(r_acc=r_acc,r_prob=r_prob,r_cm=r_cm,r_imp=r_imp,r_roc=r_roc,r_coef=r_coef)
  r <- r[sapply(r, function(x) dim(x)[1]) > 0];# remove empty slots
  return(r)
}

# Last update :"Wed Oct 18 18:11:52 2023"
# Check this function! 

#library(data.table),library(foreach),library(stringr)
mergeDB_decode<-function(mergeSet) {
  # Step1: List up all requested variables!
  if (!exists("idxF",mergeSet)){idxF <- function(i) which.min(abs(timeL-i))}
  varList<-unlist(lapply(mergeSet$fmlL,function(f){str_extract_all(f,boundary("word"))}))
  varList<-gsub('^[[:digit:]]+','',varList);#ignore numbers
  varList<-varList[!grepl(":",varList)];#ignore interaction
  varList<-gsub('_S[[:digit:]]','',varList);#TS(TimeSeg label) does not exist yet, so remove _S
  varList<-c(varList,unlist(mergeSet$exvL))#include extra supporting variables
  varList<-unique(varList[varList != ""])#remove empty elements
  tsList<-unique(unique(varList[grepl("_S^|_RSA|_pp",varList)]));#keep to-be-segmented data!
  sprintf("Loading %s & Segmenting %s",paste0(varList,collapse=","),paste0(tsList,collapse=","))
  
  # Step2: Loop through database and load variables
  dsAL<-list()
  setNumericRounding(2);# necesssary for dcast
  registerDoMC(4);
  # https://stackoverflow.com/questions/37941867/error-with-large-numerics-in-dcast-data-table
  
  for (i in 1:length(mergeSet$db)){
    sprintf("Loading data from: %s",mergeSet$db[[i]])
    res_list <- foreach(ii=1:length(mergeSet$timeS)) %dopar% {
      # Open database connection
      dbCon=dbConnect(dbDriver("SQLite"),mergeSet$db[[i]]);
      tableN<-dbListTables(dbCon)
      dbCL<-dbListFields(dbCon,tableN)
      
      # Load all relevant data of the relevant time range
      var2Load <-dbCL[dbCL %in% varList]
      tIDX<-unlist(lapply(mergeSet$timeS[[ii]], function(s){sapply(s,idxF)}))
      selectQ <- paste0('SELECT ',paste(unique(c(var2Load)),collapse = ", "))
      whereQ <- sprintf(" WHERE %s >= %d AND %s <= %d",timeV,range(tIDX)[1],timeV,range(tIDX)[2])
      querySTW<-paste0(selectQ,sprintf(" FROM %s",tableN),whereQ);print(querySTW)# For test:" LIMIT 100000"
      dsA=as.data.table(dbGetQuery(dbCon,querySTW));
      
      #Aggregate all variables (see comments for selective aggregation of variables)
      #ds<-dsA[,lapply(.SD,mean,na.rm=T),by=c(mergeSet$exvL[[1]])];
      grpV <- unique(unlist(mergeSet$exvL[1]));# everything but time and dv
      ds <- dsA[, lapply(.SD,mean,na.rm=T),by=grpV]
      tc<- tsList[tsList %in% colnames(ds)]
      setnames(ds,old=tc,new=paste0(tc,"_S",ii));
      ds[,c(timeV):=NULL]; #ds[,time:=paste(timeS[[ii]],collapse="_")];#ds[,time:=NULL]
      return(ds)
    }
    # Concatanate data # dataLL<-rbindlist(dataL)
    # # of data points differ within timge segments...(rounding ?)
    keyVars<-colnames(res_list[[1]])[!grepl("_S\\d",colnames(res_list[[1]]))]
    dsA<-Reduce(function(...) merge(...,by=keyVars),res_list)
    
    # Update Varlist (prioritize columns from earlier data base!)
    varLoaded<-gsub("_S\\d","",names(dsA))
    dsAL[i]<-list(list(dsA=dsA,varList=varList))
    varList<-unique(c(unlist(mergeSet$exvL[1:2]),varList[!varList %in% varLoaded]))
  }
  
  # Merge all the list of data set and summarize output
  rALL<-as.list(as.data.frame(do.call(rbind, dsAL)))
  dsM <- Reduce(function(...) merge(..., by=mergeSet$exvL[[1]]), rALL$dsA)
  #dsM[!which(is.nanM(dsM))];# remove NA trials...?
  dsMR <-list(dsM=dsM,varList=rALL$varList,size=unlist(lapply(rALL$dsA,function(x){dim(x)[1]})))
  return(dsMR)
}

# prewhite_decode----------------------------------------------------------------
# applies prewhitening, which is normalization of the data using the inverse noise covariance 
# specifically, 1) get residual of dummy coded predictor, 2) get residual, 
# 3) get inversr of residuals (corpcor::invcov.shrink), 4) matrix square root (pracma::sqrtm)
# output is Sigma^-1 in LDA section https://en.wikipedia.org/wiki/Linear_discriminant_analysis#LDA_for_two_classes
# [INPUT]:
# d_i: data table/data.frame containing feautres and target label
# modelV: target label
# [OUTPUT]:
# resid: rediaul or noise 
# inv_cov: inverse of noise covariance
# sq_inv_cov: matrix square root of inv_cov
# d_adj: original data feautres multiplied by sq_inv_cov
#-------------------------------------------------------------------------------
prewhite_decode <- function(d_i, modelV){
  # Get dummy codes of the target labels
  dummy <- d_i[,lapply(.SD,fastDummies::dummy_cols),.SDcols = modelV]
  d_adj <- d_i[,-modelV, with=F] # remove target label
  
  # Get residuals of all features 
  resid <- foreach::foreach(term=names(d_adj)) %dopar% {
    f <- sprintf("%s~.", term)
    r <- residuals(lm(f, data = cbind(d_i[,term,with=F], dummy[,-1])))
  } %>% bind_rows() %>% t() 
  
  # Calculate square root of the inverse of covariance matrix
  #sq_inv_cov <- pracma::sqrtm(solve(cov(resid)))$B
  inv_cov <- corpcor::invcov.shrink(resid) # with shrinkage  
  sq_inv_cov <- pracma::sqrtm(inv_cov)$B 
  d_adj <- as.matrix(d_adj) %*% sq_inv_cov
  d_adj <- cbind(d_i[, modelV, with=F], d_adj)
  
  # Put all outputs together
  results <-list(d_adj=d_adj,inv_cov=inv_cov,sq_inv_cov=sq_inv_cov,resid=resid)
  return(results)
}





# Spatial Projection----------------------------------------------------------------------------------
# m = model object from "train" function, varL = feature labels
# dd = observed data (trial, time, features)
# AMAP keeps coefficients from all classes (change this?)
# CAUTION: something is strange about coef.mda behavior!
weight2topo <- function(m,d,varL) {
  
  #Calculation of variable importance (standard method)
  i<-varImp(m, scale=T);#this is slow!
  #rownames(i$importance)<-varL;ggplot(i,top=14)
  imp<-i$importance;
  imp$ALL<-rowMeans(imp);
  imp$Vars<-varL;#rownames(i$importance)<-varL;
  imp$Freq<-gsub("_.*","",imp$Vars);
  imp$Elec<-gsub(".*_","",imp$Vars)
  
  #Weight projection
  # X = observed data (XXt is covariance)
  # W = coefficients, filter 
  # Y = component (multiply of X and W)
  w = coef(m$finalModel)
  if (any(rownames(w) %in% "Intercept")){w = w[-1,]} #<-something is odd about coef.mda.., it randomly omits "Intercept" occasionally...?
  #x <- apply(dd,c(1,3),mean)# average across time!
  x <- m$trainingData[,-1]
  
  #Method 1 (BieSman,2012;Haufe et al, 2014): A = XXtW
  imp <- cbind(imp,as.data.frame(stats::cov(x) %*% w))
  #combn(m$finalModel$obsLevels, 2)
  
  #Method 2 (Parra,2005): A = XYt (YYt)^-1
  #y = t(wm) %*% t(x);# w is individual wm is average coefs for all classes
  #imp$AMAP <- t(x) %*% t(y) %*% (y %*% t(y)) ^-1
  
  #imp %>% dplyr::arrange(desc(V2))
  
  return(imp)
}

# create_contrasts--------------------------------------------------------------
creat_contrasts <- function(modelV,rsapath,define) {
  
  # Inside of the function
  design <- list.files(path = file.path(rsapath,modelV), pattern = "Design_*",full.names =T) %>% data.table::fread()
  mlist <- list.files(path = file.path(rsapath,modelV), pattern = "_M.txt",full.names =T)
  
  # Load all RSA models and define as matrix
  mL<-lapply(as.list(mlist),function(f){read.table(f,header = F)})
  names(mL)<-gsub("*.txt","",basename(mlist)) # # List of RSA models
  if (define){for (m in 1:length(mlist)){assign(gsub(".txt","",mlist)[m],mL[[m]],envir=.GlobalEnv)}}
  
  # Code contrast
  feat<-names(design) # all task features in the design
  design[,(feat):=lapply(.SD, as.factor),.SDcols=feat]
  n_levels<-design[,lapply(.SD, n_distinct),.SDcols=feat]
  
  # Contrast functions (keep adding stuff here)
  c_nothing <- function(v){v}
  c_binary <- function(v){`contrasts<-`(v,,contr.helmert(2))}
  c_dummy <- function(v){`contrasts<-`(v,,contr.dummy(n_distinct(v)))}
  
  for (f in 1:length(feat)){
    # Change contrast functions
    cf <- c_nothing
    if (n_levels[[1,f]]==2){cf <- c_binary}
    if (n_levels[[1,f]]>2){cf <- c_dummy}
    # Assign
    design[,(feat[f]):=lapply(.SD,cf),.SDcols=feat[f]]
  }
  
  print(design)
  return(design)
}

# ====================================================================================================
# PLOTTING
# ====================================================================================================
## theme0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ggplot theme striping off everything!
theme0 <- function(...) theme( legend.position = "none",
                               panel.background = element_blank(),
                               panel.grid.major = element_blank(),
                               panel.grid.minor = element_blank(),
                               panel.spacing  = unit(0,"null"),
                               axis.ticks = element_blank(),
                               axis.text.x = element_blank(),
                               axis.text.y = element_blank(),
                               axis.title.x = element_blank(),
                               axis.title.y = element_blank(),
                               axis.ticks.length = unit(0,"null"),
                               panel.border=element_rect(color=NA),...)


## dropLeadingZero ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# remove extra leading zeros from axis labels
dropLeadingZero <- function(l){stringr::str_replace(l, '0(?=.)', '')}


## dropLeadingZero ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# force exact number of breaks for plots
equal_breaks <- function(n = 3, s = 0.05, digits=1){function(x){
  d <- s * diff(range(x)) / (1+2*s)
  dd<-seq(min(x)+d, max(x)-d, length=n)
  dd<-round(dd, digits)
}
}

## shift_axis ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# shift axis for traditional ERP plot
shift_axis <- function(p, y=0){
  g <- ggplotGrob(p)
  dummy <- data.frame(y=y)
  ax <- g[["grobs"]][g$layout$name== "axis-b"][[1]]
  p + annotation_custom(grid::grobTree(ax, vp = grid::viewport(y=1, height=sum(ax$height))), 
                        ymax=y, ymin=y) +
    geom_hline(aes(yintercept=y), data = dummy) +
    theme(axis.text.x = element_blank(),  axis.ticks.x=element_blank())
}

## plot_topo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ds = data set containing values to plot topographic pattern
# vn = variable name
prep_topo <- function (ds,vn){
  #Load chan locations
  path<-path.expand("~/Dropbox/w_ONGOINGRFILES/w_OTHERS/chanlocs_20E_Oregon.txt")
  chanlocs<-fread(paste0(path),header = TRUE);
  chanlocs$X<-scales::rescale(chanlocs$X,c(0,1));chanlocs$Y<-scales::rescale(chanlocs$Y,c(0,1));
  ds$Elec<-factor(ds$Elec);ds<-inner_join(ds,chanlocs,by="Elec")
  
  #GAM
  ds$DV <-ds[[vn]];
  splm <- gam(DV ~ s(X,Y,bs="sos",k = length(ds$Elec)-2), data=ds)
  ds_spl<-data.frame(expand.grid(X=seq(-0.3, 1.3, 0.01),Y=seq(-0.3, 1.3, 0.001)))
  ds_spl$value <- predict(splm,ds_spl, type = "response")
  
  #HEAD
  c_center=c(0.5,0.5);c_rad<-1.25;nose_y<-(c_rad/2)+c_center[1];
  circledat <- circleFun(c_center,c_rad, npoints = 100) # center on [.5, .5]
  ds_spl$incircle <- (ds_spl$X - c_center[1])^2 + (ds_spl$Y - c_center[2])^2 < (c_rad/2)^2 # mark
  ds_spl <- ds_spl[ds_spl$incircle,]
  
  return(list(ds_spl,chanlocs,circledat,nose_y,c_center,c_rad))
}

## circleFun ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#calculate xy cordinates of a specified circle (for topograpgic plot)
circleFun <- function(center = c(0,0),diameter = 1, npoints = 100){
  r = diameter / 2
  tt <- seq(0,2*pi,length.out = npoints)
  xx <- center[1] + r * cos(tt)
  yy <- center[2] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}


# ====================================================================================================
# OTHERS
# ====================================================================================================
require(tictoc)
## tic & toc ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Matlab tic & toc equivalant functions
# tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self"))
# {
#    type <- match.arg(type)
#    assign(".type", type, envir=baseenv())
#    if(gcFirst) gc(FALSE)
#    tic <- proc.time()[type]         
#    assign(".tic", tic, envir=baseenv())
#    invisible(tic)
# }
# 
# toc <- function()
# {
#    type <- get(".type", envir=baseenv())
#    toc <- proc.time()[type]
#    tic <- get(".tic", envir=baseenv())
#    print(toc - tic)
#    invisible(toc)
# }


## .ls.objects ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# improved list of objects
.ls.objects <- function (pos = 1, pattern, order.by,
                         decreasing=FALSE, head=FALSE, n=5) {
  napply <- function(names, fn) sapply(names, function(x)
    fn(get(x, pos = pos)))
  names <- ls(pos = pos, pattern = pattern)
  obj.class <- napply(names, function(x) as.character(class(x))[1])
  obj.mode <- napply(names, mode)
  obj.type <- ifelse(is.na(obj.class), obj.mode, obj.class)
  obj.size <- napply(names, object.size)
  obj.dim <- t(napply(names, function(x)
    as.numeric(dim(x))[1:2]))
  vec <- is.na(obj.dim)[, 1] & (obj.type != "function")
  obj.dim[vec, 1] <- napply(names, length)[vec]
  out <- data.frame(obj.type, obj.size, obj.dim)
  names(out) <- c("Type", "Size", "Rows", "Columns")
  if (!missing(order.by))
    out <- out[order(out[[order.by]], decreasing=decreasing), ]
  if (head)
    out <- head(out, n)
  out
}
# shorthand
lsos <- function(..., n=10) {
  .ls.objects(..., order.by="Size", decreasing=TRUE, head=TRUE, n=n)
}



# # plot_segment_across_facets~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # this draws segments across panels
# # http://rstudio-pubs-static.s3.amazonaws.com/410976_f8eb6b218bfa42038a8b7bc9a6f9a193.html
# plot_segment_across_facets <- function(p, from=1, to=2, point_id=1, 
#                                        plotout = F,
#                                        gp=gpar(lty=1, alpha=0.5)){
#   if (TRUE %in% grepl("ggplot", class(p))) {
#     g <- ggplot_gtable(ggplot_build(p))
#   } else {
#     g <- p
#   }
#   
#   # collect panel viewport names and index numbers in the grob
#   panel_vps <- c()
#   id_n <- c()
#   for (i in 1:length(g$grobs)) {
#     if (str_detect(g$layout[i, "name"], "panel") & g$grobs[[i]]$name != "NULL") {
#       p_name <- g$layout[i, "name"]
#       panel_vps <- c(panel_vps, p_name)
#       id_n <- c(id_n, i)
#     }
#   }
#   
#   # preprocessing for converting the panel #
#   panel_vps %>%
#     str_replace("panel-", "") %>%
#     str_split("[\\-\\.]") %>%
#     map_chr(1) -> ind_col
#   ind_col <- as.numeric(ind_col)
#   
#   panel_vps %>%
#     str_replace("panel-", "") %>%
#     str_split("[\\-\\.]") %>%
#     map_chr(2) -> ind_row
#   ind_row <- as.numeric(ind_row)
#   
#   my_dim <- c(max(ind_row), max(ind_col))
#   x <- 1:length(id_n)
#   L <- length(x)
#   x[(L+1):(my_dim[1]*my_dim[2])] <- NA
#   m1 <- as.vector(matrix(x, nrow=my_dim[1], byrow=T))
#   
#   x2 <- 1:L
#   xx <- as.vector(!is.na(m1))
#   xx[xx] <- x2
#   xx[!xx] <- NA
#   m2 <- as.vector(matrix(xx, nrow=my_dim[1]))
#   
#   # convert panel # to match the sequence
#   from <- m2[m1==from]
#   from <- from[complete.cases(from)]
#   to <- m2[m1==to]
#   to <- to[complete.cases(to)]
#   
#   # select points to be connected
#   pnames1 <- names(g$grobs[[id_n[from]]]$children)
#   pnames2 <- names(g$grobs[[id_n[to]]]$children)
#   
#   pname1 <- pnames1[str_detect(pnames1, "geom_point.points")]
#   pname2 <- pnames2[str_detect(pnames2, "geom_point.points")]
#   
#   p1 <- g$grobs[[id_n[from]]]$children[[pname1[1]]]
#   p2 <- g$grobs[[id_n[to]]]$children[[pname2[1]]]
#   
#   g <- with(g$layout[id_n[from],], 
#             gtable_add_grob(g, 
#                             moveToGrob(p1$x[point_id], 
#                                        p1$y[point_id]), t=t, l=l))
#   g <- with(g$layout[id_n[to],], 
#             gtable_add_grob(g, 
#                             lineToGrob(p2$x[point_id], p2$y[point_id], gp=gp), 
#                             t=t, l=l))
#   
#   g$layout$clip <- "off"
#   if (plotout==TRUE) grid.draw(g) 
#   return(g)
# }
