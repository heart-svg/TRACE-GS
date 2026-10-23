suppressPackageStartupMessages({library(data.table);library(jsonlite)})
args<-commandArgs(trailingOnly=TRUE); getarg<-function(k){i<-match(k,args);if(is.na(i))stop(k);args[i+1]}; model<-getarg('--model'); inp<-getarg('--input'); out<-getarg('--out'); dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(!model%in%c('rrBLUP','BayesA','BayesB','BayesC'))stop('model'); if(model=='rrBLUP')library(rrBLUP) else library(BGLR)
pcc<-function(a,b)cor(a,b); rows<-list(); pooled<-list()
read_gz<-function(path){
 if(!file.exists(path))stop(sprintf('missing input: %s',path))
 fread(cmd=sprintf('gzip -dc %s',shQuote(normalizePath(path))))
}
for(o in 0:4){tr<-read_gz(file.path(inp,sprintf('outer_%d_trn.csv.gz',o))); te<-read_gz(file.path(inp,sprintf('outer_%d_tst.csv.gz',o))); if(anyDuplicated(tr$ID)||anyDuplicated(te$ID)||length(intersect(tr$ID,te$ID)))stop(sprintf('ID leakage outer %d',o)); feature_cols<-setdiff(names(tr),c('ID','y_raw')); if(!identical(feature_cols,setdiff(names(te),c('ID','y_raw'))))stop(sprintf('feature mismatch outer %d',o)); X<-as.matrix(tr[,..feature_cols]); Xt<-as.matrix(te[,..feature_cols]); y<-tr$y_raw; if(any(!is.finite(X))||any(!is.finite(Xt))||any(!is.finite(y))||any(!is.finite(te$y_raw)))stop(sprintf('non-finite values outer %d',o))
 if(model=='rrBLUP'){fit<-mixed.solve(y,Z=X); pred<-as.numeric(fit$beta)+as.numeric(Xt%*%fit$u)} else {set.seed(20260902+o); fit<-BGLR(y=y,ETA=list(list(X=X,model=model)),nIter=12000,burnIn=2000,thin=5,verbose=FALSE,saveAt=file.path(out,sprintf('tmp_%d_',o))); pred<-as.numeric(fit$mu)+as.numeric(Xt%*%fit$ETA[[1]]$b)}
 rmse<-sqrt(mean((te$y_raw-pred)^2)); mae<-mean(abs(te$y_raw-pred)); r<-list(outer=o,model=model,n_train=nrow(tr),n_test=nrow(te),PCC=pcc(te$y_raw,pred),RMSE=rmse,MAE=mae); rows[[o+1]]<-r; pr<-data.frame(ID=te$ID,outer=o,y_raw=te$y_raw,pred_raw=pred); pooled[[o+1]]<-pr; fwrite(pr,file.path(out,sprintf('outer_%d_predictions_tst.csv',o))); write_json(r,file.path(out,sprintf('outer_%d_metrics.json',o)),auto_unbox=TRUE,pretty=TRUE)}
tab<-rbindlist(rows); pred<-rbindlist(pooled); if(anyDuplicated(pred$ID))stop('pooled duplicate test IDs'); fwrite(tab,file.path(out,'fivefold_metrics.csv')); fwrite(pred,file.path(out,'oof_predictions.csv')); agg<-list(status='PASS',model=model,folds=5,PCC_mean=mean(tab$PCC),PCC_sd=sd(tab$PCC),RMSE_mean=mean(tab$RMSE),MAE_mean=mean(tab$MAE),pooled_PCC=pcc(pred$y_raw,pred$pred_raw)); write_json(agg,file.path(out,'COMPLETE.json'),auto_unbox=TRUE,pretty=TRUE); print(agg)

