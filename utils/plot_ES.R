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
    npresses = json$stats$npresses,
    depth = depth,
    run = as.factor(run),
  ) %>%
    mutate(
      x = map(generations, ~json$stats$x[[.]]),
      xstar = map(generations, ~json$stats$xstar[[.]])
    )
  return(stats_tibble)
}

plot_depth <- function(dat) {
  if (nrow(dat) == 0) {
    return(ggplot() + theme_void())
  }

  # Create the facetted plot
  p1 <- ggplot(dat, aes(x = generation, y = npresses, color = run)) +
    geom_line() +
    labs(y = expression(k['max']), x = NULL) +
    scale_y_continuous(labels = function(x) paste("#", x),
                       breaks = seq(min(dat$npresses), max(dat$npresses), 1)) +
    scale_color_brewer(palette = "Paired") +
    theme(legend.position = "none")

  p2 <- dat %>%
    ggplot(aes(x = generation, y = waste, color = run)) +
    geom_line() +
    scale_y_log10(labels = function(x) { paste0(x, " m", "\u00b2") }) +  # Use custom label function
    labs(y = "Waste", x = "Generation") +
    scale_color_brewer(palette = "Paired") +
    theme(legend.position = "none")

  # Combine both plots into a single facetted plot
  plot_grid(p1, p2, nrow = 2, align = "v", rel_heights = c(0.4, 1))
}

es <- map_df(files, read_file)
es %>%
  group_by(depth) %>%
  summarise(
    runs = length(unique(run)),
    min_presses = min(npresses),
    max_presses = max(npresses),
    mu_presses = mean(npresses),
  )

plot <- plot_depth(es) + theme(plot.margin = margin(5.5, 5.5, 5.5, 5.5, "pt"))
ggsave(output_file, plot = plot, width = 3.2, height = 2.5, units = "in", dpi = 300)