# read the files from the command line
args <- commandArgs(trailingOnly = TRUE)
output_file <- args[1]
files <- args[2:length(args)]
# make sure the output file is a .png file
if (!grepl("\\.png$", output_file)) {
  stop("Output file must be a .png file")
}
source('utils/common.R')

plot_rollwidths <- function(dat, eps = 0) {
  t <- dat %>%
    mutate(max_generation = max(generation)) %>%
    filter(opt_presses == min(opt_presses)) %>%
    filter(opt_waste <= min(opt_waste) + eps) %>%
    group_by(max_generation, run) %>%
    summarise(
      generation = min(generation),
      run = list(unique(run)),
    ) %>%
    unnest(run)
  print(t)

  dat %>%
    unnest(xstar) %>%
    ggplot(aes(x = generation, y = xstar, color = run)) +
    geom_point(size = .5) +
    labs(y = expression('{' ~ rho ~ ~' }'), x = 'Generation') +
    scale_y_continuous(labels = function(x) { ifelse(x == 0, "", scales::unit_format(unit = "m", scale = 1 / 1000)(x)
    ) }) +
    geom_point(data = dat %>% merge(t) %>% unnest(xstar),
               aes(x = max_generation + nrow(t) + 1, y = xstar, color = run),
               shape = 3, size = 2,
               position = position_jitter(width = nrow(t), height = 0)
    ) +
    scale_color_brewer(palette = "Paired") +
    theme(legend.position = "none")
}

es <- read_jsons(files)
a5 <- 148 * 210 / 1e6
plot_rollwidths(es, eps = a5)
ggsave(output_file, width = 4, height = 3, units = "in", dpi = 300)