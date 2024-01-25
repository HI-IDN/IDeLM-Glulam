# Read the filename from the command line
args <- commandArgs(trailingOnly = TRUE)
file_out <- args[1]
working_dir <- args[2]
json_files <- args[3:length(args)]

source("utils/common.R")

# and the output file is a .png file
if (!grepl("\\.png$", file_out)) {
  stop("Output file must be a .png file")
}

es <- read_jsons(json_files)
es %>%
  group_by(depth) %>%
  summarise(
    runs = length(unique(run)),
    min_presses = min(opt_presses),
    max_presses = max(opt_presses),
    md_presses = median(opt_presses),
    mu_constr = mean(nconstrs),
    mu_patters = mean(npatterns),
  )

log_files <- list.files("data/slurm/", "*log", full.names = T)
log <- read_logs(log_files)

# what is the order of magnitude of the number of patterns and constraints?
pdat <- es %>%
  filter(generation <= 100) %>%
  mutate(depth = as.factor(depth), grp = interaction(run, depth)) %>%
  ggplot(aes(x = generation, y = npatterns, color = depth, group = grp)) +
  geom_line() +
  labs(x = "Generation", y = expression(n), color = 'Depth\n(mm)') +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 2))



pdur <- log %>%
  group_by(depth) %>%
  filter(type %in% c('generation', 'first_feasible')) %>%
  mutate(
    depth = as.factor(depth),
    minutes = value / 60,
    type = factor(type, levels = c('first_feasible', 'generation'),
                  labels = c('First\nFeasible', 'Total\nDuration')),
  ) %>%
  ggplot(aes(y = minutes, x = type, fill = depth)) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set1") +
  labs(x = NULL, y = 'Duration (minutes)', fill = 'Depth (mm)') +
  scale_y_sqrt() +
  theme(legend.position = "none")

p <- plot_grid(pdat, pdur, rel_widths = c(1.5, 1), ncol = 2)
ggsave(file_out, plot=p, width = 5, height = 4, units = "in", dpi = 300)