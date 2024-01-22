library(tidyverse)
library(cowplot)
theme_set(theme_minimal(base_size = 10)) # Adjust the base_size as needed

# read the files from the command line
args <- commandArgs(trailingOnly = TRUE)
output_file <- args[1]
files <- args[2:length(args)]
# make sure the output file is a .png file
if (!grepl("\\.png$", output_file)) {
  stop("Output file must be a .png file")
}

read_file <- function(file_name) {
  json <- jsonlite::fromJSON(file_name)
  depth <- json$depth
  generations <- 1:length(json$stats$gen)
  run <- gsub(paste0('soln_ES_d', depth, '_(\\d+).json[.part]*'), '\\1', basename(file_name))
  print(c(file_name, run))
  stats_tibble <- tibble(
    generation = json$stats$gen,
    waste = json$stats$waste,
    depth = depth,
    run = as.factor(run),
  ) %>%
    mutate(
      x = map(generations, ~json$stats$x[[.]]),
    ) %>%
    cbind(json$stats$run_summary)
  return(stats_tibble)
}

plot_depth <- function(dat) {
  if (nrow(dat) == 0) {
    return(ggplot() + theme_void())
  }

  # Create the facetted plot
  p1 <- ggplot(dat, aes(x = generation, y = npresses, color = run)) +
    geom_line() +
    labs(y = "Presses", x = NULL) +
    scale_y_continuous(labels = function(x) paste("#", x),
                       breaks = seq(min(dat$npresses), max(dat$npresses), 1)) +
    scale_color_brewer(palette = "Paired") +
    theme(legend.position = "none")

  p2 <- ggplot(dat, aes(x = generation, y = waste, color = run)) +
    geom_line() +
    scale_y_sqrt(labels = function(x) { paste0(x, " m", "\u00b2") }) +  # Use custom label function
    labs(y = "Waste", x = "Generation") +
    scale_color_brewer(palette = "Paired") +
    theme(legend.position = "none")

  # Combine both plots into a single facetted plot
  plot_grid(p1, p2, nrow = 2, align = "v", rel_heights = c(0.4, 1)) +
    #increase the top margin for a better title
    theme(plot.margin = margin(20, 10, 10, 10, "pt"))
}

combined_data <- map_df(files, read_file)
combined_data %>%
  group_by(depth) %>%
  summarise(
    runs = length(unique(run)),
    min_presses = min(npresses),
    max_presses = max(npresses),
    mu_presses = mean(npresses),
  )

p90a <- plot_depth(combined_data %>% filter(depth == 90))
p90b <- plot_depth(combined_data %>% filter(depth == 90))
p115 <- plot_depth(combined_data %>% filter(depth == 115))
p140 <- plot_depth(combined_data %>% filter(depth == 140))
p160 <- plot_depth(combined_data %>% filter(depth == 160))
p185 <- plot_depth(combined_data %>% filter(depth == 185))
combined_plot <- plot_grid(p90a, p90b, p115, p140, p160, p185, ncol = 2, align = "hv", rel_heights = c(1, 1),
                           labels = c("a) 90mm #1", "b) 90mm #2", "c) 115mm", "d) 140mm", "e) 160mm", "f) 185mm"),
                           label_size = 10)
ggsave(output_file, plot = combined_plot, width = 7.16, height = 8, units = "in", dpi = 300)
