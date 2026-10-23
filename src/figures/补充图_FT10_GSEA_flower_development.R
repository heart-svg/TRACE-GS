#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3) stop("Usage: Rscript plot_supp_ft10_gsea_baseR.R <source.csv> <outdir> <resultsdir>")
input <- args[1]; outdir <- args[2]; resultsdir <- args[3]
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)
dir.create(resultsdir, recursive=TRUE, showWarnings=FALSE)
d <- read.csv(input, check.names=FALSE, stringsAsFactors=FALSE, fileEncoding="UTF-8-BOM")
stopifnot(all(c("rank","rank_metric","running_ES","hit","label","reported_NES","reported_FDR") %in% names(d)))

blue <- "#90A4C4"; coral <- "#F4B7AD"; grey <- "#D4D4D4"; charcoal <- "#444444"

draw_gsea <- function() {
  layout(matrix(1:3,ncol=1), heights=c(2.4,0.55,1.05))
  op <- par(family="sans", las=1, mgp=c(2.0,0.55,0), tcl=-0.2, cex.axis=0.72, cex.lab=0.78)
  on.exit(par(op), add=TRUE)
  boxes <- list(); capture <- function(id){f<-par("fig");p<-par("plt");c(id=id,x0=f[1]+p[1]*(f[2]-f[1]),y0=f[3]+p[3]*(f[4]-f[3]),x1=f[1]+p[2]*(f[2]-f[1]),y1=f[3]+p[4]*(f[4]-f[3]))}

  par(mar=c(0.6,5.0,2.1,0.8))
  plot(d$rank,d$running_ES,type="l",lwd=1.5,col=blue,xaxt="n",xlab="",ylab="Running ES")
  abline(h=0,col="#888888",lty=2,lwd=0.6)
  k <- which.min(d$running_ES)
  points(d$rank[k],d$running_ES[k],pch=21,bg=coral,col="white",cex=1.15)
  title(main=sprintf("FT10: positive regulation of flower development (GO:0009911)\nNES = %.2f; FDR = %.3g", d$reported_NES[1], d$reported_FDR[1]), cex.main=0.78, font.main=2)
  mtext("a",side=3,adj=-0.10,line=0.45,font=2,cex=1.0)
  boxes[[1]] <- capture("a")

  par(mar=c(0.1,5.0,0.1,0.8))
  plot(c(1,nrow(d)*1.18),c(0,1),type="n",axes=FALSE,xlab="",ylab="")
  hit_idx <- which(d$hit %in% c(TRUE,"True","TRUE"))
  segments(hit_idx,0.10,hit_idx,0.90,col=charcoal,lwd=0.35)
  segments(1,0.5,nrow(d),0.5,col=grey,lwd=0.4)
  labs <- which(nchar(d$label)>0)
  label_order <- match(c("FT","TSF","FUL","SOC1"), d$label[labs])
  label_order <- label_order[!is.na(label_order)]
  labs <- labs[label_order]
  ypos <- seq(0.88,0.22,length.out=length(labs))
  xlabpos <- round(nrow(d)*1.035)
  # Orthogonal leaders prevent one gene's diagonal connector from crossing
  # another gene label in the compact barcode panel.
  segments(d$rank[labs],0.50,d$rank[labs],ypos,col="#A0A0A0",lwd=0.45)
  segments(d$rank[labs],ypos,xlabpos-round(nrow(d)*0.040),ypos,col="#A0A0A0",lwd=0.45)
  text(xlabpos,ypos,d$label[labs],pos=4,offset=0.30,cex=0.62,font=2,
       col=ifelse(d$label[labs]%in%c("FT","SOC1"),"#617AA5","#C77F75"),xpd=NA)
  mtext("b",side=3,adj=-0.10,line=-0.65,font=2,cex=1.0)
  boxes[[2]] <- capture("b")

  par(mar=c(3.0,5.0,0.5,0.8))
  colfill <- ifelse(d$rank_metric>=0,blue,coral)
  plot(d$rank,d$rank_metric,type="n",xlab="Rank in ordered expression-feature list",ylab="")
  polygon(c(d$rank,rev(d$rank)),c(d$rank_metric,rep(0,nrow(d))),col=adjustcolor(grey,alpha.f=0.62),border=NA)
  lines(d$rank,d$rank_metric,col="#777777",lwd=0.45)
  abline(h=0,col="#888888",lwd=0.5)
  mtext("Rank metric",side=2,line=3.7,las=0,cex=0.78)
  mtext("c",side=3,adj=0.00,line=0.85,font=2,cex=1.0)
  boxes[[3]] <- capture("c")
  boxes
}

prefix <- file.path(outdir,"supp_ft10_gsea_flower_development")
width_in <- 89/25.4; height_in <- 105/25.4
pdf(paste0(prefix,".pdf"),width=width_in,height=height_in,family="Helvetica",useDingbats=FALSE); boxes<-draw_gsea(); dev.off()
svg(paste0(prefix,".svg"),width=width_in,height=height_in,family="sans"); draw_gsea(); dev.off()
png(paste0(prefix,".png"),width=width_in,height=height_in,units="in",res=300,type="cairo"); draw_gsea(); dev.off()
tiff(paste0(prefix,".tiff"),width=width_in,height=height_in,units="in",res=600,compression="lzw",type="cairo"); draw_gsea(); dev.off()

ptw<-width_in*72; pth<-height_in*72
panels<-lapply(seq_along(boxes),function(i){b<-boxes[[i]];list(id=as.character(b["id"]),bbox_pt=as.numeric(c(b["x0"],b["y0"],b["x1"],b["y1"]))*c(ptw,pth,ptw,pth),grid_id="r-base-layout",row_start=i,row_stop=i,col_start=1,col_stop=1)})
manifest<-list(schema_version=1,backend="r-base",figure=list(width_pt=ptw,height_pt=pth),panels=panels,
               exemptions=list(list(panels=c("a","b","c"),checks=c("vertical-gutter"),reason="Standard GSEA evidence stack uses intentionally unequal running-score, hit-barcode and rank-metric panel gutters")))
jsonlite::write_json(manifest,paste0(prefix,".alignment-layout.json"),auto_unbox=TRUE,pretty=TRUE)
write.csv(d,file.path(resultsdir,"supp_ft10_gsea_source_copy.csv"),row.names=FALSE,fileEncoding="UTF-8")
cat(prefix,"\n")

