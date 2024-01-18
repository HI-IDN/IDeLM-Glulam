library(tidyverse)
library(cowplot)
theme_set(theme_minimal(base_size = 10)) # Adjust the base_size as needed

read_file <- function(file_name) {
  print(file_name)
  json <- jsonlite::fromJSON(file_name)
  depth <- json$depth
  stats_tibble <- data.frame(
    generation = json$stats$gen,
    #xstar = I(json$stats$xstar),
    #sstar = I(json$stats$sstar),
    #sucstar = I(json$stats$sucstar),
    waste = json$stats$waste,
    npresses = json$stats$npresses,
    x = I(json$stats$x),
    sigma = I(json$stats$sigma),
    depth = depth,
    run = file_name
    #run_summary = ifelse(is.null(json$stats$run_summary), NULL, I(json$stats$run_summary)),
  )
}

plot_depth <- function(run_depth) {
  dat <- combined_data %>% filter(depth == run_depth)

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

files <- list.files("data/v1.0/", pattern = "\\.json|\\.json.part$", full.names = TRUE)
combined_data <- map_df(files, read_file)
combined_data %>%
  group_by(depth) %>%
  summarise(
    runs = length(unique(run)),
    min_presses = min(npresses),
    max_presses = max(npresses),
  )

p90a <- ggplot()
p90b <- ggplot()
p115 <- plot_depth(115)
p140 <- plot_depth(140)
p160 <- plot_depth(160)
p185 <- plot_depth(185)
combined_plot <- plot_grid(p90a, p90b, p115, p140, p160, p185, ncol = 2, align = "hv", rel_heights = c(1, 1),
                           labels = c("a) 90mm #1", "b) 90mm #2", "c) 115mm", "d) 140mm", "e) 160mm", "f) 185mm"),
                           label_size = 10)
ggsave("data/v1.0/ES_run.png", plot = combined_plot, width = 7.16, height = 8, units = "in", dpi = 300)

