#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) stop("Usage: Rscript plot_figure4_go_gsea_baseR.R <summary.csv> <ora.csv> <gsea.csv> <outdir> <resultsdir>")
summary_csv <- args[1]; ora_csv <- args[2]; gsea_csv <- args[3]; outdir <- args[4]; resultsdir <- args[5]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(resultsdir, recursive = TRUE, showWarnings = FALSE)

read_utf8 <- function(path) read.csv(path, check.names = FALSE, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")
d <- read_utf8(summary_csv); ora <- read_utf8(ora_csv); gsea <- read_utf8(gsea_csv)

trait_map <- c("FT10"="FT10", "FT16"="FT16", "Diameter"="Diameter", "100GW"="100-grain weight",
               "Heading"="Heading date", "Fruit fly"="Fly starvation", "Stickleback"="Stickleback immunity")
trait_order <- unname(trait_map)
d$trait_full <- unname(trait_map[d$trait])

cols <- c("positive"="#90A4C4", "weak"="#F4B7AD", "negative"="#A9A9A9")
trait_cols <- c("FT10"="#90A4C4", "FT16"="#7F94B7", "Diameter"="#D4D4D4",
                "100-grain weight"="#F4B7AD", "Heading date"="#D9978D",
                "Fly starvation"="#A89BB3", "Stickleback immunity"="#8CB7AA")

wrap_term <- function(x, width=30) vapply(x, function(z) paste(strwrap(z, width=width), collapse="\n"), character(1))
short_trait <- c("100-grain weight"="100GW", "Heading date"="Heading", "Fly starvation"="Fly", "Stickleback immunity"="Fish")
shorten_trait <- function(x) ifelse(x %in% names(short_trait), unname(short_trait[x]), x)

draw_figure <- function() {
  layout(matrix(1:3, nrow=1), widths=c(0.90, 1.25, 1.30))
  op <- par(family="sans", las=1, mgp=c(2.1,0.55,0), tcl=-0.22, cex.axis=0.70, cex.lab=0.78,
            mar=c(3.2,3.3,1.0,0.8), xaxs="i", yaxs="i")
  on.exit(par(op), add=TRUE)
  bboxes <- list()
  capture_bbox <- function(id) {
    fig <- par("fig"); plt <- par("plt")
    c(id=id,
      x0=fig[1] + plt[1]*(fig[2]-fig[1]),
      y0=fig[3] + plt[3]*(fig[4]-fig[3]),
      x1=fig[1] + plt[2]*(fig[2]-fig[1]),
      y1=fig[3] + plt[4]*(fig[4]-fig[3]))
  }

  # a, stable genes versus paired OOF RNA gain.
  xr <- c(min(d$bootstrap_ci95_low)-0.015, max(d$bootstrap_ci95_high)+0.035)
  yr <- c(0, max(d$stable_genes)*1.12)
  plot(d$pooled_delta_pcc, d$stable_genes, type="n", xlim=xr, ylim=yr,
       xlab="Pooled outer-fold OOF Delta PCC", ylab="Cross-fold stable RNA genes")
  abline(v=0, lty=2, col="#777777", lwd=0.7)
  segments(d$bootstrap_ci95_low, d$stable_genes, d$bootstrap_ci95_high, d$stable_genes,
           col=cols[d$gain_class], lwd=1.2)
  points(d$pooled_delta_pcc, d$stable_genes, pch=21, bg=cols[d$gain_class], col="white", lwd=0.7, cex=1.35)
  # Keep the negative-gain Diameter label inside the panel; only the
  # rightmost stickleback point is labelled to the left.
  lab_pos <- c(4,4,4,4,4,4,2)
  label_x <- d$pooled_delta_pcc
  label_x[d$trait_full == "Diameter"] <- 0.012
  text(label_x, d$stable_genes + 4.2,
       labels=shorten_trait(d$trait_full), pos=lab_pos, offset=0.32,
       cex=0.56, col="#444444")
  mtext("a", side=3, adj=-0.13, line=-0.5, font=2, cex=1.05)
  bboxes[[1]] <- capture_bbox("a")

  # b, significant ORA representatives after semantic reduction.
  par(mar=c(3.2,8.7,1.0,0.8))
  if (nrow(ora) == 0) {
    plot.new(); text(0.5,0.5,"No FDR-significant ORA terms")
  } else {
    ora <- ora[order(match(ora$trait, trait_order), ora$fdr), ]
    yy <- rev(seq_len(nrow(ora)))
    labels <- paste0(shorten_trait(ora$trait), " | ", wrap_term(ora$display_term, 27))
    maxr <- max(ora$gene_ratio)*1.20
    plot(ora$gene_ratio, yy, type="n", yaxt="n", xlim=c(0,maxr), ylim=c(0.4,nrow(ora)+0.6),
         xlab="Gene ratio", ylab="")
    axis(2, at=yy, labels=labels, las=1, tick=FALSE, cex.axis=0.55, line=-0.25)
    fdr_score <- pmin(8, ora$minus_log10_fdr)
    pal <- colorRampPalette(c("#D4D4D4", "#90A4C4", "#617AA5"))(101)
    ci <- pmax(1, pmin(101, 1+round(fdr_score/8*100)))
    size <- 0.75 + 1.9*sqrt(ora$gene_count/max(ora$gene_count))
    points(ora$gene_ratio, yy, pch=21, bg=pal[ci], col="#5B5B5B", cex=size, lwd=0.45)
    # Marker size is defined in the caption; an in-panel key would obscure
    # the lowest terms at final journal size.
  }
  mtext("b", side=3, adj=-0.52, line=-0.5, font=2, cex=1.05)
  bboxes[[2]] <- capture_bbox("b")

  # c, direction and magnitude of representative GSEA modules.
  par(mar=c(3.2,9.2,1.0,0.9))
  gsea <- gsea[order(match(gsea$trait, trait_order), -abs(gsea$NES)), ]
  yy <- rev(seq_len(nrow(gsea)))
  labels <- paste0(shorten_trait(gsea$trait), " | ", wrap_term(gsea$display_term, 28))
  lim <- max(abs(gsea$NES))*1.18
  plot(gsea$NES, yy, type="n", yaxt="n", xlim=c(-lim,lim), ylim=c(0.4,nrow(gsea)+0.6),
       xlab="Normalized enrichment score (NES)", ylab="")
  abline(v=0, col="#777777", lwd=0.65)
  axis(2, at=yy, labels=labels, las=1, tick=FALSE, cex.axis=0.55, line=-0.25)
  fdr_score <- pmin(8, gsea$minus_log10_fdr)
  pospal <- colorRampPalette(c("#D4D4D4", "#90A4C4", "#617AA5"))(101)
  negpal <- colorRampPalette(c("#D4D4D4", "#F4B7AD", "#D98F83"))(101)
  ci <- pmax(1, pmin(101, 1+round(fdr_score/8*100)))
  fill <- ifelse(gsea$NES >= 0, pospal[ci], negpal[ci])
  size <- 0.75 + 1.9*sqrt(gsea$gene_count/max(gsea$gene_count))
  segments(0, yy, gsea$NES, yy, col="#C7C7C7", lwd=0.8)
  points(gsea$NES, yy, pch=21, bg=fill, col="#5B5B5B", cex=size, lwd=0.45)
  mtext("c", side=3, adj=-0.92, line=-0.5, font=2, cex=1.05)
  bboxes[[3]] <- capture_bbox("c")
  bboxes
}

prefix <- file.path(outdir, "figure4_go_gsea_semantic_reduced")
width_in <- 183/25.4; height_in <- 104/25.4
pdf(paste0(prefix,".pdf"), width=width_in, height=height_in, family="Helvetica", useDingbats=FALSE)
bbox <- draw_figure(); dev.off()
svg(paste0(prefix,".svg"), width=width_in, height=height_in, family="sans"); draw_figure(); dev.off()
png(paste0(prefix,".png"), width=width_in, height=height_in, units="in", res=300, type="cairo"); draw_figure(); dev.off()
tiff(paste0(prefix,".tiff"), width=width_in, height=height_in, units="in", res=600, compression="lzw", type="cairo"); draw_figure(); dev.off()

# Write exact plot-area geometry for backend-neutral alignment auditing.
ptw <- width_in*72; pth <- height_in*72
panels <- lapply(seq_along(bbox), function(i) {
  b <- bbox[[i]]
  list(id=as.character(b["id"]), bbox_pt=as.numeric(c(b["x0"],b["y0"],b["x1"],b["y1"]))*c(ptw,pth,ptw,pth),
       grid_id="r-base-layout", row_start=1, row_stop=1, col_start=i, col_stop=i)
})
manifest <- list(schema_version=1, backend="r-base", figure=list(width_pt=ptw,height_pt=pth), panels=panels,
                 exemptions=list(list(panels=c("a","b","c"), checks=c("panel-width","horizontal-gutter"), reason="Different inferential roles and left-label budgets require intentional unequal panel widths and gutters")))
jsonlite::write_json(manifest, paste0(prefix,".alignment-layout.json"), auto_unbox=TRUE, pretty=TRUE)
write.csv(d, file.path(resultsdir,"figure4a_stability_gain.csv"), row.names=FALSE, fileEncoding="UTF-8")
cat(prefix,"\n")

