# HAL2017_PUSHMORE_L1Model_RSA_PTSRCONJ
# load pred file (contains single-trial accuracy & confidence)
# fit RSA models to decoding resulf of the conjunction variable
# More appropriate CTR for PTSRCONJ (use truly conjunction case specific RT)
# _pred could be from: CROSS,JMB,SEG, and DYN!
# =============================================================
#Removing evrything from workspace
graphics.off()
rm(list = ls(all = TRUE))

#Setting up directory
fsep<-.Platform$file.sep;
Dir_R<-path.expand("~/Dropbox/w_ONGOINGRFILES/w_OTHERS")
Dir_EDATA<-path.expand("~/Dropbox/w_SCRIPTS/P.HAL2017/HAL2017_PUSH/DataAnalysis/Matlab/EEG")
Dir_BDATA<-path.expand("~/Dropbox/w_SCRIPTS/P.HAL2017/HAL2017_PUSH/DataAnalysis/Matlab/BEH/w_ALLGRAND")
Dir_GRAND<-paste0(Dir_EDATA,"/w_ALLGRAND")
Dir_RSAMODEL<-paste0(Dir_EDATA,"/w_RSAMODELS")

# Load libraries
library(data.table)
library(tidyverse)
library(broom)
library(rhdf5)
library(caret)
library(foreach)
library(doMC)
library(binhf)
library(RColorBrewer)
library(RSQLite)
library(lazyeval)
library(stringr)
library(psych)
library(lme4)
library(rio)

# Source Files
setwd(Dir_R)
source('basic_lib.R')

# Setting for Database
setwd(Dir_EDATA);
subs<-list.files( pattern="^A[[:digit:]]{3}");
modelV<-c("PTSRCONJ","")#TSCONJ/L1TSCONJ/TCTSCONJ/L1TSCONJ_TSCONJ
fileN<-"HAL2017_PUSHMORE"
f2Load<-paste0(modelV[1],'_nbG*._pred.rds');#
# f2Load<-paste0(modelV[1],'_nbG*._RL_pred_DYN.rds');#DYN
dbName<-paste0("FIT_RSA_",modelV[1],"_CTRR");
saveOn<-T;
dataL<-list();

# # Load beh data
setwd(Dir_BDATA)
ds_beh<-fread("HAL2017_PUSHMORE_BehP.txt");
ds_beh[is.nanM(ds_beh)]<-NA
ds_beh[, (c("TASKTS","RSIJIT","EV_BLOCK","EV_TRIAL","EV_CUE","EV_STIM","EV_RESP")):=NULL]
# ds_beh<-merge(ds_beh,fread(list.files(pattern="*.BehAddVars.txt")))#Load beh file with additional vars

# # Load Individuals' Conjunction RT
ds_cj<-fread(sprintf("HAL2017_PUSHMORE_%s_CONTROL_GRP.txt",modelV[1]))

#======================================================================================================
#Load RSA Models (adjusted for different decoding variables)
#======================================================================================================
# # Adjust RSA model sources!
setwd(paste0(Dir_RSAMODEL,"/",modelV[1]))
mlist <- list.files(pattern = ".txt")
mL<-lapply(as.list(mlist),function(f){read.table(f,header = F)})
names(mL)<-gsub("*.txt","",mlist) # # List of RSA models
for (m in 1:length(mlist)){assign(gsub(".txt","",mlist)[m],mL[[m]],envir=.GlobalEnv)}

#======================================================================================================
#Process3- Database
#======================================================================================================
# Open Database  (erase old database of the same datatype, if it exists!!)
setwd(Dir_GRAND);
registerDoMC(8)
resultsA<-list();
if (file.exists(paste0(fileN,'_',dbName))){file.remove(list.files(pattern=dbName))}
dbCon=dbConnect(dbDriver("SQLite"),paste0(fileN,'_',dbName))# Open database connection
s<-28;ccc<-1

for (s in 1:(length(subs))){#1:(length(subs))
  # Load data & set properties
  Dir_Data_i<-paste0(Dir_EDATA,fsep,subs[s],fsep,"DATASETS");
  setwd(Dir_Data_i);f2LoadL<-list.files(pattern=paste0("*_",f2Load),full.names=TRUE);print(f2LoadL)
  pred<-data.table(bind_rows(lapply(f2LoadL, readRDS),.id="PGRP"));
  pred$PGRP<-as.numeric(pred$PGRP);#specific for V,H,C,CC rule experiment (or PTSRCONJ)
  subN <- gsub('[[:alpha:]]+', '', subs[s])
  mesVs<-colnames(pred)[colnames(pred) %in% LETTERS_ex(50)]
  idVs<-colnames(pred)[!colnames(pred) %in% c(mesVs,"PGRP","acc","SUBID")]
  nmbVs<-idVs[!idVs %in% c("FBAND","ELEC","CV")]# avoid character columns in idVs 
  timeVs<-nmbVs[grepl('time',nmbVs)]# time coding columns
  # Casting happen after summarizing PGRP grouping (so ignore PGRP)
  castF1<-paste0("CLASS~",paste0(idVs[!idVs %in% c("SUBID_S")],collapse="+"))
  castF2<-paste0(paste0(idVs,collapse="+"),"~vars")
  
  # Step1: Melt data from wide (A:L) to long format (BLOCK,TRIAL,time)
  pred<-data.table::melt(pred,#this is way faster than gather/spread
                         id.vars = c("PGRP",idVs),
                         measure.vars=mesVs,
                         variable.name="CLASS",value.name="PP")
  
  # Step1':Logit transform
  pred[PP==0,PP:=1e-10];pred[PP==1,PP:=1-1e-10]
  pred$PP<-psych::logit(pred$PP);
  
  # Check if all classes have complete cases 
  if (any(c(is.infinite(pred$PP) | is.nan(pred$PP)| is.na(pred$PP)))){
    pred<-pred[!c(is.infinite(pred$PP) | is.nan(pred$PP) | is.na(pred$PP)),]
    pred[,cmp:=uniqueN(CLASS),by=c("BLOCK","TRIAL",timeVs)]
    pred<-pred[cmp==length(mesVs),];pred[,cmp:=NULL]
  }
  
  # Step2: Add condition of Modeled variable from behavior (if necessary!)
  if (!any(grepl("obs",names(pred)))){
    ds_behi<-ds_beh[SUBID_S==subN,c(c("SUBID_S","BLOCK","TRIAL"),modelV[1]),with=F]
    pred<-merge(pred,ds_behi,by=c("SUBID_S","BLOCK","TRIAL"));setnames(pred,modelV[1],"obs")
  }
  
  # Step3-1:Long format (BLOCK,TRIAL,time) to another wide format (A:L to all combination of BLOCK,TRIAL,time)
  # Then, add model terms (e.g.,TASK,STIM,RESP,CONJ)
  if(is.factor(pred$obs)){pred[,obs:=as.numeric(obs)]}
  clist<-which(mesVs %in% LETTERS_ex(50))
  clistC<-expand.grid(unique(pred$PGRP),clist)#specific for V,H,C,CC rule experiment (or PTSRCONJ)
  
  # Step (4): Prepare control vector of RT
  ds_cji <- ds_cj[SUBID_S==subN,] %>% arrange("PTSRGROUP",modelV[1])
  ds_cji[,ACC:=scale(ACC),by=PTSRGROUP];ds_cji[,RT:=scale(RT),by=PTSRGROUP]
  print(ds_cji)

  results<-
    foreach(ccc=1:dim(clistC)[1]) %dopar% {
      # # Subset condition
      d<-pred[PGRP==clistC[ccc,1] & obs==clistC[ccc,2]]
      predF<-data.table::dcast(d,castF1, value.var="PP")
      
      # PTSRCONJ (Rule X Resp(2)/Stim(2) with limited context) with RT control
      #predF$RULE_RSA<-RULE_M[,clistC[ccc,2]];#RULE MODEL
      #predF$STIM_RSA<-STIMPOS_M[,clistC[ccc,2]];#STIM MODEL
      #predF$RESP_RSA<-RESP_M[,clistC[ccc,2]];#RESP MODEL
      #predF$SRMAP_RSA<-SRMAP_M[,clistC[ccc,2]];#SRMAP MODEL
      #predF$RSRMAP_RSA<-RSRMAP_M[,clistC[ccc,2]];#Rule-specific SRMAP MODEL
      #predF$CTRRT_RSA<-ds_cji[PTSRGROUP==clistC[ccc,1],]$RT;#CONTROL RT MODEL
      ##predF$CTRACC_RSA<-ds_cji[PTSRGROUP==clistC[ccc,1],]$ACC;#CONTROL MODEL
      
      # PTSRCCONJ (Cue X Rule X Resp(2)/Stim(2) with limited context) with RT control
      predF$CUE_RSA<-CUE_M[,clistC[ccc,2]];#RULE MODEL
      predF$RULE_RSA<-RULE_M[,clistC[ccc,2]];#RULE MODEL
      predF$STIM_RSA<-STIMPOS_M[,clistC[ccc,2]];#STIM MODEL
      predF$RESP_RSA<-RESP_M[,clistC[ccc,2]];#RESP MODEL
      predF$SRMAP_RSA<-SRMAP_M[,clistC[ccc,2]];#SRMAP MODEL
      predF$RSRMAP_RSA<-RSRMAP_M[,clistC[ccc,2]];#Rule-specific SRMAP MODEL
      predF$CRSRMAP_RSA<-CRSRMAP_M[,clistC[ccc,2]];#Rule-specific SRMAP MODEL
      predF$CTRRT_RSA<-ds_cji[PTSRGROUP==clistC[ccc,1],]$RT;#CONTROL RT MODEL
      ##predF$CTRACC_RSA<-ds_cji[PTSRGROUP==clistC[ccc,1],]$ACC;#CONTROL MODEL
      
      # Step3-2:Regression all at once!
      dvIdx<-grepl("CLASS|*._RSA$",names(predF))
      Y<-as.matrix(predF[,which(!dvIdx),with=FALSE]);#DVs(this indexing takes time...)
      X<-as.matrix(predF[,which(dvIdx)[-1],with=FALSE]);#IVs(first var is CLASS!)
      # r<-coeff(.lm.fit(cbind(1,X),Y));#fastest, but only gives unstandardized coefficients
      r<-ls.print(lsfit(X,Y),print.it=F);#includes intercept!
      
      # Step3-3:Summarize results!
      names(r$coef.table)<-colnames(predF)[!dvIdx]
      estList<-rownames(r$coef.table[[1]]);#list of vars for estimates
      r<-rbindlist(lapply(r$coef.table,as.data.frame),idcol=TRUE)
      setnames(r,old=c("t-value","Pr(>|t|)","Std.Err"),new=c("tvalue","pvalue","SE"))
      
      #Rename variables and add basic identifier variables
      r[,c("SUBID_S",modelV[1]):=list(pred$SUBID_S[1],clistC[ccc,2])]#specific for V,H,C,CC rule experiment (or PTSRCONJ)
      r[,vars:=rep(estList,dim(r)[1]/length(estList))]
      
      #Restore labels from list names, then convert to wide format
      bltrtime<-str_split_fixed(r$.id,"_",n=Inf);#no assumption for .id tokens
      r[,c(idVs[!idVs %in% c("SUBID","SUBID_S")]):=narray::split(bltrtime,along=2)]# assume SUBID is the first column
      r[,(nmbVs):=lapply(.SD,as.numeric),.SDcols=nmbVs]
      r<-r[vars!="Intercept",c(idVs,"vars","tvalue"),with=F]
      
      # Need wide format(dcast) // long-wide(dcast) & wide-long(melt)
      r_wide<-dcast(r,castF2,value.var="tvalue")
      return(r_wide)
    }
  
  # Summarize all results!!!!
  results <- rbindlist(results)
  setorderv(results,nmbVs)
  depV<-colnames(results)[!colnames(results) %in% c(idVs,"CTRACC_RSA","CTRRT_RSA","ODDEVEN_RSA")]
  
  # Detect outliers within each time point and replace with NA
  for (cc in depV){
    results[,std:=lapply(.SD,sd),by=c(timeVs),.SDcols=cc]
    results[,bad:=abs(results[[cc]]) > std*5] # outlier based on std
    results[,(cc):=ifelse(bad,NA,results[[cc]])]
    print(sprintf("For %s, %f2 percent of rows were removed",cc,(length(which(results$bad))/dim(results)[1])*100))
    results<-results[bad!=T,] # remove bad trials altogether??
    #results[,bad:=outlier(results[[cc]],logical=T)] # outlier based on package(only one value??)
  }
  
  # Merge to behavioral template and put into DB
  results[,c("std","bad"):=NULL]# reduce data!
  setnames(results,depV, paste0(depV,modelV[2])) #Rename F1(High TestP) and F2(Low TestP)
  predB<-merge(results,ds_beh,by=c("SUBID_S","BLOCK","TRIAL"));#use ds_beh(containing all vars)
  dbWriteTable(dbCon,name=dbName,value=predB,row.names=FALSE,append=TRUE);#Try to append!
}

# Checking
# dscheck=as.data.table(dbGetQuery(dbCon,paste0('SELECT * FROM ',dbName,'  LIMIT 5')));
