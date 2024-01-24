library(xtable)
source("utils/common.R")

raw <- read.csv('data/glulam.csv')
raw %>% glimpse()
raw %>%
  group_by(depth) %>%
  summarise(
    n_beams = n(),
    n_order = n_distinct(order),
    n_quantity = sum(quantity),
    area_sqm = sum(width * height * quantity / 1e6),
    kmax_lb = as.integer(floor(area_sqm / area_press)),
  ) %>%
  arrange(depth) %>%
  xtable(type = "latex", caption = "Summary of Glulam Beams Production Data") %>%
  print(include.rownames = FALSE, booktabs = TRUE)


log_files <- list.files("data/slurm/", "*log", full.names = T)
log <- map_df(log_files, read_log)
log %>%
  filter(type %in% (c('total_runtime', 'first_feasible', 'generation'))) %>%
  group_by(depth) %>%
  mutate(hours = value / 60^2, minutes = value / 60, generation = generation + 1) %>%
  summarise(
    runs = n_distinct(run),
    md_gen = as.integer(median(generation[type == 'total_runtime'])),
    mu_dur = mean(hours[type == 'total_runtime']),
    lb_dur = quantile(hours[type == 'total_runtime'], 0.05),
    ub_dur = quantile(hours[type == 'total_runtime'], 0.95),
    mu_fea = mean(minutes[type == 'first_feasible']),
    mx_fea = max(minutes[type == 'first_feasible']),
    mu_gendur = mean(minutes[type == 'generation']),
    lb_gendur = quantile(minutes[type == 'generation'], 0.05),
    ub_gendur = quantile(minutes[type == 'generation'], 0.95),
  ) %>%
  merge(
    log %>%
      filter(type %in% c("presses", "presses_not_full")) %>%
      group_by(depth, run, generation) %>%
      mutate(prev_presses = lag(value)) %>%
      filter(type == "presses_not_full") %>%
      rename(presses_not_full = value) %>%
      group_by(depth, presses_not_full) %>%
      summarise(md_initpresses = as.integer(median(prev_presses))) %>%
      ungroup() %>%
      select(depth, md_initpresses)
  ) %>%
  arrange(depth) %>%
  xtable(type = "latex", caption = "Summary of Glulam Beams Optimization Runs") %>%
  print(include.rownames = FALSE, booktabs = TRUE)
