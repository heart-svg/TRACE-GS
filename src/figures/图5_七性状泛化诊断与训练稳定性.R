library(ggplot2)
library(patchwork)
library(jsonlite)
library(svglite)
library(ragg)
source("panel_alignment.R")

root <- "/path/to/trace_gs_workspace"
input_dir <- file.path(root, "paper_completion_20261015/generalization_summary")
stage <- file.path(root, "figure5_staging")
out_fig <- file.path(stage, "figures")
out_res <- file.path(stage, "results")
out_note <- file.path(stage, "notes")
out_qa <- file.path(stage, "qa")
dir.create(out_fig, recursive = TRUE, showWarnings = FALSE)
dir.create(out_res, recursive = TRUE, showWarnings = FALSE)
dir.create(out_note, recursive = TRUE, showWarnings = FALSE)
dir.create(out_qa, recursive = TRUE, showWarnings = FALSE)

diag <- read.csv(file.path(input_dir, "figure5_diagnostic_source_data.csv"), check.names = FALSE)
learn <- read.csv(file.path(input_dir, "figure5_learning_source_data.csv"), check.names = FALSE)
stopifnot(nrow(diag) == 7L, nrow(learn) == 35L)
stopifnot(all(table(learn$trait_key) == 5L))

trait_order <- c("FT10", "FT16", "Diam.", "100GW", "Heading", "Fly", "Fish")
diag$label <- factor(diag$label, levels = trait_order)
learn$label <- factor(learn$label, levels = trait_order)

# Palette fixed for the manuscript: low-saturation grey, coral and blue-grey.
pal <- c(grey = "#D4D4D4", coral = "#F4B7AD", blue = "#90A4C4",
         purple = "#A89AB8", teal = "#91B7B0", ink = "#30343B")

base_theme <- theme_classic(base_size = 7.2, base_family = "sans") +
  theme(
    axis.line = element_line(linewidth = 0.35, colour = pal["ink"]),
    axis.ticks = element_line(linewidth = 0.35, colour = pal["ink"]),
    axis.text = element_text(size = 6.2, colour = pal["ink"]),
    axis.title = element_text(size = 7.0, colour = pal["ink"]),
    plot.title = element_text(size = 7.4, face = "bold", hjust = 0, colour = pal["ink"]),
    plot.subtitle = element_text(size = 5.8, colour = "#666666", margin = margin(b = 3)),
    legend.title = element_blank(),
    legend.text = element_text(size = 6.0),
    legend.key.size = grid::unit(3.5, "mm"),
    panel.grid.major.y = element_line(linewidth = 0.22, colour = "#ECECEC"),
    panel.grid.minor = element_blank(),
    plot.margin = margin(4, 5, 4, 5)
  )

# a, pooled OOF predictive correlation.
p_a <- ggplot(diag, aes(label, pcc)) +
  geom_segment(aes(xend = label, y = 0, yend = pcc), linewidth = 1.7,
               colour = pal["blue"], lineend = "butt") +
  geom_point(shape = 21, size = 2.5, stroke = 0.45, fill = "white", colour = pal["ink"]) +
  scale_y_continuous(limits = c(0, 0.86), breaks = seq(0, 0.8, 0.2), expand = c(0, 0)) +
  labs(title = "Pooled outer-test prediction", subtitle = "TRACE-GS DNA+RNA; one OOF prediction per sample",
       x = NULL, y = "PCC") + base_theme +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, vjust = 1))

# b, errors normalized by phenotype SD; lower is better.
err <- rbind(
  data.frame(label = diag$label, metric = "NRMSE", value = diag$nrmse),
  data.frame(label = diag$label, metric = "NMAE", value = diag$nmae)
)
err$metric <- factor(err$metric, levels = c("NRMSE", "NMAE"))
p_b <- ggplot(err, aes(label, value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.68), width = 0.58,
           colour = "white", linewidth = 0.25) +
  scale_fill_manual(values = c(NRMSE = unname(pal["blue"]), NMAE = unname(pal["coral"]))) +
  scale_y_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
  labs(title = "Scale-normalized prediction error", subtitle = "NRMSE (blue-grey) and NMAE (coral), divided by phenotype SD",
       x = NULL, y = "Normalized error") + base_theme +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, vjust = 1),
        legend.position = "none")

# c, residual-versus-prediction slope; zero denotes no linear residual trend.
diag$slope_class <- ifelse(abs(diag$residual_slope) <= 0.10, "Near zero", "Residual trend")
p_c <- ggplot(diag, aes(label, residual_slope, fill = slope_class)) +
  geom_hline(yintercept = 0, linewidth = 0.45, linetype = 2, colour = "#777777") +
  geom_col(width = 0.62, colour = "white", linewidth = 0.25) +
  geom_text(aes(label = sprintf("%.2f", residual_slope)),
            vjust = ifelse(diag$residual_slope >= 0, -0.55, 1.45),
            size = 2.05, family = "sans", colour = pal["ink"]) +
  scale_fill_manual(values = c("Near zero" = unname(pal["blue"]), "Residual trend" = unname(pal["coral"]))) +
  scale_y_continuous(limits = c(-0.59, 0.13), breaks = c(-0.5, -0.25, 0, 0.1), expand = c(0, 0)) +
  labs(title = "Residual calibration diagnostic", subtitle = "Slope of OOF residuals against OOF predictions",
       x = NULL, y = "Residual slope") + base_theme +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, vjust = 1),
        legend.position = "none")

# d, fold-wise best early-stopping epochs. Log scale preserves the true Fish outer2=150 value.
epoch_summary <- aggregate(best_epoch ~ label, learn, median)
p_d <- ggplot(learn, aes(label, best_epoch)) +
  geom_line(aes(group = outer_fold), colour = pal["grey"], linewidth = 0.3, alpha = 0.8) +
  geom_point(aes(fill = factor(outer_fold)), shape = 21, size = 2.05, stroke = 0.35,
             colour = pal["ink"], position = position_jitter(width = 0.07, height = 0, seed = 20261018)) +
  geom_crossbar(data = epoch_summary, aes(x = label, y = best_epoch, ymin = best_epoch, ymax = best_epoch),
                width = 0.52, linewidth = 0.65, colour = pal["ink"], inherit.aes = FALSE) +
  scale_fill_manual(values = c("0" = unname(pal["blue"]), "1" = unname(pal["coral"]), "2" = unname(pal["teal"]),
                               "3" = unname(pal["purple"]), "4" = "#B7A58B"), guide = "none") +
  scale_y_log10(breaks = c(1, 3, 10, 30, 100, 150), labels = c("1", "3", "10", "30", "100", "150"),
                limits = c(0.85, 190)) +
  labs(title = "Fold-wise early-stopping behavior", subtitle = "Five outer folds; horizontal bars show medians",
       x = NULL, y = "Best epoch (log scale)") + base_theme +
  theme(axis.text.x = element_text(angle = 40, hjust = 1, vjust = 1))

fig <- wrap_plots(p_a, p_b, p_c, p_d, ncol = 2, byrow = TRUE) +
  plot_annotation(tag_levels = "a", tag_prefix = "", tag_suffix = "") &
  theme(plot.tag = element_text(family = "sans", face = "bold", size = 9, colour = pal["ink"]),
        plot.tag.position = c(0.01, 0.99))

width_mm <- 183
height_mm <- 132
width_in <- width_mm / 25.4
height_in <- height_mm / 25.4
base <- file.path(out_fig, "Figure5_generalization_and_training_stability")

# Exact plotted source data and untouched source copies.
write.csv(diag, file.path(out_res, "Figure5_generalization_source_data.csv"), row.names = FALSE)
write.csv(learn, file.path(out_res, "Figure5_best_epoch_source_data.csv"), row.names = FALSE)
write.csv(err, file.path(out_res, "Figure5_normalized_error_plot_data.csv"), row.names = FALSE)

# Patchwork geometry is a symmetric 2 x 2 grid. Preserve an auditable manifest.
manifest <- list(
  figure = basename(base), backend = "R/ggplot2/patchwork",
  final_size_mm = c(width_mm, height_mm), tolerance_pt = 1.5,
  panels = list(
    list(id = "a", row = 1, col = 1, role = "pooled OOF prediction"),
    list(id = "b", row = 1, col = 2, role = "normalized error"),
    list(id = "c", row = 2, col = 1, role = "residual calibration"),
    list(id = "d", row = 2, col = 2, role = "early stopping stability")
  ),
  comparison_groups = list(rows = list(c("a", "b"), c("c", "d")), columns = list(c("a", "c"), c("b", "d")))
)
write_json(manifest, file.path(out_qa, "Figure5_layout_manifest.json"), pretty = TRUE, auto_unbox = TRUE)

require_patchwork_panel_alignment(
  plot = fig,
  manifest_path = file.path(out_qa, "Figure5_alignment-layout.json"),
  report_path = file.path(out_qa, "Figure5_alignment.json"),
  overlay_svg = file.path(out_qa, "Figure5_alignment.svg"),
  width_in = width_in,
  height_in = height_in,
  panel_ids = c("a", "b", "c", "d"),
  row_groups = list(c("a", "b"), c("c", "d")),
  column_groups = list(c("a", "c"), c("b", "d")),
  audit_script = "/path/to/codex_home/skills/nature-figure/scripts/audit_panel_alignment.py",
  python = "/path/to/runtime_cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  strict = TRUE
)

svglite(paste0(base, ".svg"), width = width_in, height = height_in, bg = "white")
print(fig); dev.off()
grDevices::pdf(paste0(base, ".pdf"), width = width_in, height = height_in,
               family = "Helvetica", bg = "white", useDingbats = FALSE)
print(fig); dev.off()
agg_tiff(paste0(base, ".tiff"), width = width_in, height = height_in, units = "in", res = 600,
         background = "white", compression = "lzw")
print(fig); dev.off()
agg_png(paste0(base, ".png"), width = width_in, height = height_in, units = "in", res = 300,
        background = "white")
print(fig); dev.off()

message("Wrote Figure 5 bundle to: ", base)


