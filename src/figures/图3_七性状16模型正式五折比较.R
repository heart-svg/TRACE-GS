#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript plot_model_comparison.R <summary.csv> <output_dir> <results_dir> <panel_alignment.R>")
}
input_csv <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(args[[2]], mustWork = TRUE)
results_dir <- normalizePath(args[[3]], mustWork = TRUE)
alignment_helper <- normalizePath(args[[4]], mustWork = TRUE)
source(alignment_helper)

raw <- read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)
required <- c("dataset", "model", "n", "valid_PCC_folds", "mean_fold_PCC", "sd_fold_PCC")
stopifnot(all(required %in% names(raw)))
stopifnot(nrow(raw) > 0L)

dataset_map <- data.frame(
  dataset = c(
    "Arabidopsis_FT10", "Arabidopsis_FT16", "Arabidopsis_Diameter",
    "Maize_100grainweight", "Maize_Headingdate",
    "Drosophila_StarvationRes_M", "Stickleback_A_Freq.Lymph"
  ),
  trait = c("FT10", "FT16", "Diameter", "100-grain weight", "Heading date", "Fly starvation", "Stickleback immunity"),
  panel = c("a  Arabidopsis thaliana", "a  Arabidopsis thaliana", "a  Arabidopsis thaliana",
            "b  Zea mays", "b  Zea mays", "c  Animal validation", "c  Animal validation"),
  trait_order = c(1, 2, 3, 1, 2, 1, 2),
  stringsAsFactors = FALSE
)

model_order <- c(
  "TRACE-GS DNA+RNA", "TRACE-GS DNA-only",
  "DEM DNA+RNA", "DEM-Fix DNA+RNA", "DEM DNA-only", "DEM-Fix DNA-only",
  "rrBLUP", "BayesA", "BayesB", "BayesC", "LASSO",
  "SVR", "XGBoost", "DeepGS", "LCNN", "Cropformer"
)

model_label <- c(
  "TRACE-GS DNA+RNA" = "TRACE +RNA",
  "TRACE-GS DNA-only" = "TRACE DNA",
  "DEM DNA+RNA" = "DEM +RNA",
  "DEM-Fix DNA+RNA" = "DEM-Fix +RNA",
  "DEM DNA-only" = "DEM DNA",
  "DEM-Fix DNA-only" = "DEM-Fix DNA",
  "rrBLUP" = "rrBLUP", "BayesA" = "BayesA", "BayesB" = "BayesB",
  "BayesC" = "BayesC", "LASSO" = "LASSO", "SVR" = "SVR",
  "XGBoost" = "XGBoost", "DeepGS" = "DeepGS", "LCNN" = "LCNN",
  "Cropformer" = "Cropformer"
)

palette <- c(
  "TRACE-GS DNA+RNA" = "#D98276", "TRACE-GS DNA-only" = "#90A4C4",
  "DEM DNA+RNA" = "#E8A398", "DEM-Fix DNA+RNA" = "#F4B7AD",
  "DEM DNA-only" = "#AEB6BC", "DEM-Fix DNA-only" = "#D4D4D4",
  "rrBLUP" = "#7189AD", "BayesA" = "#879CBD", "BayesB" = "#9DAECC",
  "BayesC" = "#B4C1DA", "LASSO" = "#CAD4E4",
  "SVR" = "#AD96B3", "XGBoost" = "#C2ACC6",
  "DeepGS" = "#7CA99B", "LCNN" = "#9ABEB2", "Cropformer" = "#BCD4CC"
)

dat <- merge(raw, dataset_map, by = "dataset", all = FALSE)
dat <- dat[dat$model %in% model_order, ]
stopifnot(nrow(dat) == nrow(dataset_map) * length(model_order))
stopifnot(all(is.finite(dat$mean_fold_PCC[!is.na(dat$mean_fold_PCC)])))
stopifnot(all(is.finite(dat$sd_fold_PCC[!is.na(dat$sd_fold_PCC)])))
dat$not_computable <- is.na(dat$mean_fold_PCC)
dat$plot_pcc <- ifelse(dat$not_computable, 0, dat$mean_fold_PCC)
dat$plot_sd <- ifelse(dat$not_computable | is.na(dat$sd_fold_PCC), 0, dat$sd_fold_PCC)
dat$model <- factor(dat$model, levels = model_order)
dat$trait <- factor(dat$trait, levels = dataset_map$trait[order(match(dataset_map$panel, unique(dataset_map$panel)), dataset_map$trait_order)])

# Record the exact plotted values as source data.
dat <- dat[order(dat$panel, dat$trait_order, dat$model), ]
source_data <- dat[, c("panel", "trait", "dataset", "model", "n", "valid_PCC_folds", "mean_fold_PCC", "sd_fold_PCC")]
write.csv(source_data, file.path(results_dir, "figure3_model_comparison_source_data.csv"), row.names = FALSE, fileEncoding = "UTF-8")

theme_set(theme_classic(base_size = 7.2, base_family = "sans"))

make_panel <- function(panel_name, letter) {
  d <- dat[dat$panel == panel_name, ]
  # Label the proposed fused model and the strongest comparator for each trait.
  best_other <- do.call(rbind, lapply(split(d[d$model != "TRACE-GS DNA+RNA" & !d$not_computable, ], d[d$model != "TRACE-GS DNA+RNA" & !d$not_computable, ]$trait), function(z) {
    z[which.max(z$mean_fold_PCC), , drop = FALSE]
  }))
  trace_rows <- d[d$model == "TRACE-GS DNA+RNA", , drop = FALSE]
  labels <- unique(rbind(trace_rows, best_other))

  ggplot(d, aes(x = trait, y = plot_pcc, fill = model)) +
    geom_hline(yintercept = 0, linewidth = 0.25, colour = "#555555") +
    geom_col(position = position_dodge2(width = 0.88, preserve = "single", padding = 0.08),
             width = 0.82, colour = "white", linewidth = 0.12) +
    geom_errorbar(aes(ymin = plot_pcc - plot_sd, ymax = plot_pcc + plot_sd),
                  position = position_dodge2(width = 0.88, preserve = "single", padding = 0.08),
                  width = 0.12, linewidth = 0.25, colour = "#444444") +
    geom_text(data = labels,
              aes(label = sprintf("%.2f", mean_fold_PCC), y = mean_fold_PCC + sd_fold_PCC + 0.035),
              position = position_dodge2(width = 0.88, preserve = "single", padding = 0.08),
              size = 2.0, family = "sans", colour = "#222222", vjust = 0) +
    geom_text(data = d[d$not_computable, , drop = FALSE], aes(label = "NC", y = -0.075),
              position = position_dodge2(width = 0.88, preserve = "single", padding = 0.08),
              size = 2.0, family = "sans", colour = "#555555", angle = 0, vjust = 0.5) +
    scale_fill_manual(values = palette, labels = model_label, drop = FALSE) +
    scale_y_continuous(limits = c(-0.18, 1.02), breaks = seq(-0.1, 1.0, 0.2), expand = expansion(mult = c(0, 0))) +
    labs(x = NULL, y = "Mean outer-fold PCC (+/- SD)", tag = letter) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.title.y = element_text(size = 7.2, margin = margin(r = 4)),
      axis.text.x = element_text(size = 7.0, face = "bold", margin = margin(t = 3)),
      axis.text.y = element_text(size = 6.5),
      legend.position = "top",
      plot.tag = element_text(size = 9.5, face = "bold"),
      plot.tag.position = c(0.005, 0.985),
      plot.margin = margin(5, 4, 4, 5),
      panel.grid = element_blank()
    )
}

p_a <- make_panel("a  Arabidopsis thaliana", "a")
p_b <- make_panel("b  Zea mays", "b")
p_c <- make_panel("c  Animal validation", "c")

combined <- (p_a / p_b / p_c) +
  plot_layout(heights = c(1.25, 1.0, 1.0), guides = "collect") &
  theme(legend.position = "top", legend.text = element_text(size = 6.0),
        legend.key.size = grid::unit(3.0, "mm"), legend.spacing.x = grid::unit(0.8, "mm"),
        legend.box.margin = margin(0, 0, 1, 0)) &
  guides(fill = guide_legend(nrow = 3, byrow = TRUE, title = NULL))

prefix <- file.path(output_dir, "figure3_seven_trait_model_comparison")
width_mm <- 183
height_mm <- 200
width_in <- width_mm / 25.4
height_in <- height_mm / 25.4

require_patchwork_panel_alignment(
  combined,
  manifest_path = paste0(prefix, ".alignment-layout.json"),
  report_path = paste0(prefix, ".alignment.json"),
  overlay_svg = paste0(prefix, ".alignment.svg"),
  width_in = width_in,
  height_in = height_in,
  tolerance_pt = 1.5,
  gutter_tolerance_pt = 1.5,
  strict = TRUE,
  audit_script = file.path(dirname(alignment_helper), "audit_panel_alignment.py"),
  python = Sys.getenv("PYTHON", unset = Sys.which("python"))
)

svglite::svglite(paste0(prefix, ".svg"), width = width_in, height = height_in)
print(combined)
dev.off()
# Base pdf is used instead of cairo_pdf because it preserves separately auditable
# editable text objects on this Windows R build.
grDevices::pdf(paste0(prefix, ".pdf"), width = width_in, height = height_in,
               family = "Helvetica", useDingbats = FALSE)
print(combined)
dev.off()
ragg::agg_tiff(paste0(prefix, ".tiff"), width = width_in, height = height_in, units = "in", res = 600, compression = "lzw")
print(combined)
dev.off()
ragg::agg_png(paste0(prefix, ".png"), width = width_in, height = height_in, units = "in", res = 300)
print(combined)
dev.off()

cat(prefix, "\n")

