# Read the filename from the command line
args <- commandArgs(trailingOnly = TRUE)
file_in <- args[1]
file_out <- args[2]

source("utils/common.R")
library(ggpattern)

# Make sure the file exists and is a .csv file
if (!file.exists(file_in) || !grepl("\\.csv$", file_in)) {
  stop("File does not exist or is not a .csv file")
}
# and the output file is a .png file
if (!grepl("\\.png$", file_out)) {
  stop("Output file must be a .png file")
}

plot_press <- function(file_in, file_out = "") {
  press <- read_csv(file_in, show_col_types = FALSE)
  area <- press %>%
    group_by(type) %>%
    mutate(h = h * 0.045, w = w / 1e3, area = h * w) %>%
    summarise(area = sum(area))
  items <- press %>%
    filter(type == 'item') %>%
    mutate(item = as.factor(as.numeric(sub_type) + 1))
  buffer <- press %>% filter(type == 'buffer')

  total_area <- area %>% filter(type == 'Lp') %>% pull(area)
  used_area <- area %>% filter(type == 'item') %>% pull(area)
  info <- paste0("Depth: ", gsub(".*_d(\\d+).*", "\\1", file_in), "mm, ",
                 "Presses: ", max(press$k) + 1, ', ',
                 "Items: ", length(levels(items$item)), ', ',
                 "Area: ", round(total_area, 2), "m\u00b2, ",
                 "Waste:", round(total_area - used_area, 2), "m\u00b2")

  press_block <- press %>% filter(type == 'Lp')
  plot <- ggplot(data = items, aes(x = x, y = y)) +
    # Waste
    geom_rect_pattern(
      data = press_block,
      aes(xmin = x, xmax = x + w, ymin = y, ymax = y + h),
      pattern_color = "red", pattern_fill = "red", pattern = "stripe",
      pattern_angle = 0, pattern_density = 1, pattern_spacing = 0.01,
      fill = NA, color = NA,
    ) +
    # helper lines that are the constraints
    geom_hline(yintercept = c(11, 24, 26), linetype = "dashed", color = "gray") +
    geom_vline(xintercept = 25000 - 16000, linetype = "dashed", color = "gray") +
    # Items
    geom_rect(
      aes(xmin = x, xmax = x + w, ymin = y, ymax = y + h, fill = item),
      color = 'black', linewidth = 0.1,
    ) +
    # Buffer
    geom_rect_pattern(
      data = buffer,
      aes(xmin = x, xmax = x + w, ymin = y, ymax = y + h),
      pattern_color = "black", pattern_fill = "black",
      fill = NA, color = 'black', linewidth = 0.1,
    ) +
    # annotate where the regions are
    geom_text(data = press_block, aes(x = x, y = y + h, label = paste0('R[', r, ']')), parse = TRUE,
              size = 2, hjust = 1.1, vjust = 1) +
    scale_fill_viridis_d(name = 'Item', guide = "none") +
    facet_wrap(~k, labeller = as_labeller(function(value) { paste("Press #", as.numeric(value) + 1) })) +
    scale_y_continuous(labels = function(x) paste("#", x)) +
    scale_x_continuous(labels = function(x) { ifelse(x == 0, "", scales::unit_format(unit = "m", scale = 1 / 1000)(x)
    ) },
                       limits = c(-100, 25000)) +
    labs(caption = info, x = NULL, y = "Layers") + # Update axis labels
    theme(legend.position = "bottom")  # Move legends below the plot

  if (file_out != "")
  { ggsave(file = file_out, plot = plot, width = 7.16, height = 3, units = "in", dpi = 300) }
  else
  { print(plot) }
}

plot_press(file_in, file_out)