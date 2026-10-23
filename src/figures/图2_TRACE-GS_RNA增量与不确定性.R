#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(svglite)
  library(ragg)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript plot_figure2_trace_gain.R <paired_statistics.csv> <output_dir> <results_dir> <panel_alignment.R>")
}
input_csv <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(args[[2]], mustWork = TRUE)
results_dir <- normalizePath(args[[3]], mustWork = TRUE)
alignment_helper <- normalizePath(args[[4]], mustWork = TRUE)
source(alignment_helper)

d <- read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)
required <- c("trait", "pooled_dna_pcc", "pooled_fusion_pcc", "pooled_delta_pcc",
              "bootstrap_ci95_low", "bootstrap_ci95_high", "randomization_q_two_sided_BH")
stopifnot(all(required %in% names(d)), nrow(d) == 7L)

trait_order <- c("Arabidopsis_FT10", "Arabidopsis_FT16", "Arabidopsis_Diameter",
                 "Maize_100grainweight", "Maize_Headingdate",
                 "Stickleback_A_Freq.Lymph", "Drosophila_StarvationRes_M")
trait_labels <- c(
  Arabidopsis_FT10 = "Arabidopsis FT10",
  Arabidopsis_FT16 = "Arabidopsis FT16",
  Arabidopsis_Diameter = "Arabidopsis Diameter",
  Maize_100grainweight = "Maize 100-grain weight",
  Maize_Headingdate = "Maize Heading date",
  Stickleback_A_Freq.Lymph = "Stickleback lymphocyte frequency",
  Drosophila_StarvationRes_M = "Drosophila male starvation resistance"
)
d <- d[match(trait_order, d$trait), ]
d$trait_label <- factor(unname(trait_labels[d$trait]), levels = rev(unname(trait_labels[trait_order])))
d$class <- ifelse(d$pooled_delta_pcc < 0, "Negative trend",
                  ifelse(d$pooled_delta_pcc > 0 & d$randomization_q_two_sided_BH < 0.05,
                         "FDR-supported gain", "Weak/uncertain"))
d$class <- factor(d$class, levels = c("FDR-supported gain", "Weak/uncertain", "Negative trend"))
d$q_label <- paste0("q=", formatC(d$randomization_q_two_sided_BH, format = "fg", digits = 3))

source_data <- d[, c("trait", "trait_label", "pooled_dna_pcc", "pooled_fusion_pcc",
                     "pooled_delta_pcc", "bootstrap_ci95_low", "bootstrap_ci95_high",
                     "randomization_q_two_sided_BH", "class")]
write.csv(source_data, file.path(results_dir, "figure2_trace_gain_source_data.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

pal <- c(grey = "#D4D4D4", coral = "#F4B7AD", blue = "#90A4C4", ink = "#30343B")
class_cols <- c("FDR-supported gain" = unname(pal["blue"]),
                "Weak/uncertain" = unname(pal["grey"]),
                "Negative trend" = unname(pal["coral"]))

base_theme <- theme_classic(base_size = 8.0, base_family = "sans") +
  theme(axis.line = element_line(linewidth = 0.35, colour = pal["ink"]),
        axis.ticks = element_line(linewidth = 0.35, colour = pal["ink"]),
        axis.text = element_text(size = 6.8, colour = pal["ink"]),
        axis.title = element_text(size = 7.6, colour = pal["ink"]),
        plot.tag = element_text(size = 10.5, face = "bold"),
        plot.tag.position = c(0.01, 0.99),
        panel.grid = element_blank(), plot.margin = margin(6, 8, 6, 6))

long <- rbind(
  data.frame(trait_label = d$trait_label, model = "DNA-only", PCC = d$pooled_dna_pcc),
  data.frame(trait_label = d$trait_label, model = "DNA+RNA", PCC = d$pooled_fusion_pcc)
)
long$model <- factor(long$model, levels = c("DNA-only", "DNA+RNA"))

p_a <- ggplot() +
  geom_segment(data = d, aes(y = trait_label, yend = trait_label,
                             x = pooled_dna_pcc, xend = pooled_fusion_pcc),
               colour = "#B8B8B8", linewidth = 0.85) +
  geom_point(data = long, aes(x = PCC, y = trait_label, fill = model),
             shape = 21, size = 3.0, stroke = 0.25, colour = "white") +
  scale_fill_manual(values = c("DNA-only" = unname(pal["blue"]),
                               "DNA+RNA" = unname(pal["coral"]))) +
  scale_x_continuous(limits = c(0.1, 0.83), breaks = seq(0.2, 0.8, 0.1)) +
  labs(tag = "A", x = "Pooled outer-test PCC", y = NULL, fill = NULL) +
  base_theme + theme(legend.position = c(0.86, 0.10),
                     legend.text = element_text(size = 6.8),
                     legend.key.size = grid::unit(3.5, "mm"),
                     axis.line.y = element_blank(), axis.ticks.y = element_blank())

p_b <- ggplot(d, aes(x = pooled_delta_pcc, y = trait_label, colour = class)) +
  geom_vline(xintercept = 0, linetype = 2, linewidth = 0.45, colour = "#666666") +
  geom_errorbar(aes(xmin = bootstrap_ci95_low, xmax = bootstrap_ci95_high),
                orientation = "y", width = 0, linewidth = 0.70, colour = "#666666") +
  geom_point(size = 3.2) +
  geom_text(aes(x = 0.475, label = q_label),
            hjust = 0, size = 2.5, colour = pal["ink"], show.legend = FALSE) +
  scale_colour_manual(values = class_cols, drop = FALSE) +
  scale_x_continuous(limits = c(-0.08, 0.60), breaks = seq(0, 0.4, 0.1)) +
  labs(tag = "B", x = "RNA gain in pooled PCC (95% paired bootstrap CI)", y = NULL) +
  base_theme + theme(legend.position = "none", axis.text.y = element_blank(),
                     axis.line.y = element_blank(), axis.ticks.y = element_blank())

combined <- p_a | p_b
prefix <- file.path(output_dir, "figure2_trace_gain_muted_palette")
width_mm <- 183; height_mm <- 82
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

svglite::svglite(paste0(prefix, ".svg"), width = width_in, height = height_in); print(combined); dev.off()
grDevices::pdf(paste0(prefix, ".pdf"), width = width_in, height = height_in,
               family = "Helvetica", useDingbats = FALSE); print(combined); dev.off()
grDevices::cairo_pdf(paste0(prefix, "_cairo.pdf"), width = width_in, height = height_in,
                     family = "Arial"); print(combined); dev.off()
ragg::agg_tiff(paste0(prefix, ".tiff"), width = width_in, height = height_in,
               units = "in", res = 600, compression = "lzw"); print(combined); dev.off()
ragg::agg_png(paste0(prefix, ".png"), width = width_in, height = height_in,
              units = "in", res = 300); print(combined); dev.off()

cat(prefix, "\n")

