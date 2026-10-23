#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("Usage: Rscript plot_structure_cv_sensitivity.R <source.csv> <output_dir> <results_dir> <panel_alignment.R>")
input_csv <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(args[[2]], mustWork = TRUE)
results_dir <- normalizePath(args[[3]], mustWork = TRUE)
alignment_helper <- normalizePath(args[[4]], mustWork = TRUE)
source(alignment_helper)

d <- read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)
required <- c("species", "trait", "random_delta", "random_ci_low", "random_ci_high",
              "structure_delta", "structure_ci_low", "structure_ci_high",
              "structure_positive_folds", "interpretation")
stopifnot(all(required %in% names(d)), nrow(d) == 6L)
numeric_cols <- c("random_delta", "random_ci_low", "random_ci_high", "structure_delta",
                  "structure_ci_low", "structure_ci_high", "structure_positive_folds")
stopifnot(all(vapply(d[numeric_cols], function(x) all(is.finite(x)), logical(1))))

d$trait_label <- c("FT10", "FT16", "Diameter", "100-grain weight", "Heading date", "Fly starvation")[match(
  d$trait, c("FT10", "FT16", "Diameter", "100-grain weight", "Heading date", "Male starvation resistance"))]
d$category <- factor(d$interpretation,
                     levels = c("stable", "uncertain", "fallback"),
                     labels = c("Stable positive gain", "Uncertain", "DNA-only fallback"))
d$positive_label <- paste0(d$structure_positive_folds, "/5")
d$trait_shape <- factor(d$trait_label,
                        levels = c("FT10", "FT16", "Diameter", "100-grain weight", "Heading date", "Fly starvation"))
d$label_x <- c(0.010, 0.035, -0.057, -0.020, 0.058, 0.106)[match(
  d$trait_label, c("FT10", "FT16", "Diameter", "100-grain weight", "Heading date", "Fly starvation"))]
d$label_y <- c(0.052, 0.083, -0.014, 0.026, 0.066, 0.025)[match(
  d$trait_label, c("FT10", "FT16", "Diameter", "100-grain weight", "Heading date", "Fly starvation"))]

write.csv(d, file.path(results_dir, "figure6_structure_cv_source_data.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

cols <- c("Stable positive gain"="#90A4C4", "Uncertain"="#F4B7AD", "DNA-only fallback"="#A9A9A9")
shapes <- c("Stable positive gain"=16, "Uncertain"=17, "DNA-only fallback"=18)

theme_set(theme_classic(base_size = 8.2, base_family = "sans"))

p_a <- ggplot(d, aes(x = random_delta, y = structure_delta, colour = category, shape = trait_shape)) +
  geom_hline(yintercept = 0, colour = "#B8B8B8", linewidth = 0.35) +
  geom_vline(xintercept = 0, colour = "#B8B8B8", linewidth = 0.35) +
  geom_abline(slope = 1, intercept = 0, colour = "#9B9B9B", linewidth = 0.40, linetype = 2) +
  geom_errorbar(aes(ymin = structure_ci_low, ymax = structure_ci_high), width = 0,
                linewidth = 0.55, alpha = 0.95) +
  geom_errorbar(aes(xmin = random_ci_low, xmax = random_ci_high), orientation = "y", width = 0,
                linewidth = 0.55, alpha = 0.95) +
  geom_point(size = 3.2, stroke = 0.6) +
  scale_colour_manual(values = cols, drop = FALSE) +
  scale_shape_manual(values = c(16, 17, 15, 18, 3, 4), guide = "none") +
  scale_x_continuous(limits = c(-0.07, 0.16), breaks = seq(-0.05, 0.15, 0.05)) +
  scale_y_continuous(limits = c(-0.07, 0.15), breaks = seq(-0.05, 0.15, 0.05)) +
  labs(tag = "A", x = "RNA gain under random CV (pooled Delta PCC)",
       y = "RNA gain under structure-grouped CV\n(pooled Delta PCC)") +
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 7), legend.key.width = grid::unit(4, "mm"),
        plot.tag = element_text(size = 11, face = "bold"),
        plot.tag.position = c(0.01, 0.99),
        axis.title = element_text(size = 8), axis.text = element_text(size = 7),
        plot.margin = margin(7, 8, 6, 7))

forest_order <- c("FT16", "Heading date", "FT10", "Fly starvation", "100-grain weight", "Diameter")
d$forest_trait <- factor(d$trait_label, levels = rev(forest_order))
p_b <- ggplot(d, aes(x = structure_delta, y = forest_trait, colour = category, shape = trait_shape)) +
  geom_vline(xintercept = 0, colour = "#555555", linewidth = 0.45) +
  geom_errorbar(aes(xmin = structure_ci_low, xmax = structure_ci_high), orientation = "y", width = 0,
                linewidth = 0.75) +
  geom_point(size = 3.2) +
  geom_text(aes(x = 0.165, label = positive_label), colour = "#555555", size = 2.8,
            hjust = 1, show.legend = FALSE) +
  scale_colour_manual(values = cols, drop = FALSE) +
  scale_shape_manual(values = c(16, 17, 15, 18, 3, 4), guide = "none") +
  scale_x_continuous(limits = c(-0.115, 0.172), breaks = seq(-0.10, 0.15, 0.05)) +
  labs(tag = "B", subtitle = "Positive folds shown at right",
       x = "Structure-grouped RNA gain (pooled Delta PCC; 95% CI)", y = NULL) +
  theme(legend.position = "none", plot.tag = element_text(size = 11, face = "bold"),
        plot.tag.position = c(0.01, 0.99), plot.subtitle = element_text(size = 7, hjust = 1, colour = "#555555"),
        axis.title.x = element_text(size = 8, margin = margin(t = 8)), axis.text = element_text(size = 7),
        axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = margin(7, 7, 6, 10))

combined <- p_a | p_b
prefix <- file.path(output_dir, "figure6_structure_cv_sensitivity")
width_mm <- 183; height_mm <- 100
width_in <- width_mm / 25.4; height_in <- height_mm / 25.4

require_patchwork_panel_alignment(
  combined,
  manifest_path = paste0(prefix, ".alignment-layout.json"),
  report_path = paste0(prefix, ".alignment.json"),
  overlay_svg = paste0(prefix, ".alignment.svg"),
  width_in = width_in, height_in = height_in,
  tolerance_pt = 1.5, gutter_tolerance_pt = 1.5, strict = TRUE,
  audit_script = file.path(dirname(alignment_helper), "audit_panel_alignment.py"),
  python = Sys.getenv("PYTHON", unset = Sys.which("python"))
)

svglite::svglite(paste0(prefix, ".svg"), width = width_in, height = height_in)
print(combined); dev.off()
grDevices::pdf(paste0(prefix, ".pdf"), width = width_in, height = height_in,
               family = "Helvetica", useDingbats = FALSE)
print(combined); dev.off()
grDevices::cairo_pdf(paste0(prefix, "_cairo.pdf"), width = width_in, height = height_in,
                     family = "Arial")
print(combined); dev.off()
ragg::agg_tiff(paste0(prefix, ".tiff"), width = width_in, height = height_in,
               units = "in", res = 600, compression = "lzw")
print(combined); dev.off()
ragg::agg_png(paste0(prefix, ".png"), width = width_in, height = height_in,
              units = "in", res = 300)
print(combined); dev.off()

cat(prefix, "\n")

