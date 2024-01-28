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

subsets <- list.files('data/subset/', '*.csv', full.names = T)[c(1:5, 10)] %>%
  lapply(read.csv) %>%
  bind_rows() %>%
  group_by(depth) %>%
  summarise(
    m = n(),
    sum_b = sum(quantity),
    area_sqm = sum(width * height * quantity / 1e6),
    kmax_lb = as.integer(floor(area_sqm / area_press)),
  ) %>%
  ungroup() %>%
  mutate(depth = factor(depth, levels = c(90, 91, 115, 140, 160, 185),
                        labels = c('90 (1)', '90 (2)', '115', '140', '160', '185')))


es_files <- list.files("data/v1.2/", "*json*", full.names = T)
es <- read_jsons(es_files)


log_files <- list.files("data/slurm/", "*log", full.names = T)
log <- read_logs(log_files)
stats <- log %>%
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
    lb_fea = quantile(minutes[type == 'first_feasible'], 0.05),
    ub_fea = quantile(minutes[type == 'first_feasible'], 0.95),
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
  merge(subsets) %>%
  merge(
    es %>%
      mutate(s_max = map_dbl(xstar, length)) %>%
      group_by(depth, run) %>%
      filter(generation == max(generation)) %>% # only consider the last generation
      group_by(depth) %>%
      summarise(
        runs = n_distinct(run),
        md_constr = as.integer(median(nconstrs)),
        md_nvars = as.integer(median(nvars)),
        mu_smax = mean(s_max),
        lb_smax = quantile(s_max, 0.05),
        ub_smax = quantile(s_max, 0.95),
        kmax_star = min(opt_presses),
        waste_star = min(opt_waste[opt_presses == kmax_star]),
        pct_kmaxstar = as.integer(100*mean(opt_presses == kmax_star)),
        pct_waste_a5 = as.integer(100*mean(opt_waste <= waste_star + a5 & opt_presses == kmax_star)),
        pct_waste_a4 = as.integer(100*mean(opt_waste <= waste_star + a4 & opt_presses == kmax_star)),
        min_waste = min(opt_waste),
        max_waste = max(opt_waste),
        max_presses = max(opt_presses),
      )
  ) %>%
  arrange(depth) %>%
  ungroup()

stats %>%
  select(depth, m, sum_b, runs, md_constr, md_nvars, mu_dur, lb_dur, ub_dur, md_gen, mu_fea, lb_fea, ub_fea, mu_gendur,
         lb_gendur, ub_gendur) %>%
  xtable(type = "latex", caption = "Summary of Glulam Beams Optimization Run Duration") %>%
  print(include.rownames = FALSE, booktabs = TRUE)


stats %>%
  select(depth, runs, kmax_lb,
         kmax_star, waste_star,
         max_presses, min_waste, max_waste,
         pct_kmaxstar, pct_waste_a5, pct_waste_a4,
         mu_smax, lb_smax, ub_smax
  ) %>%
  xtable(type = "latex", caption = "Summary of Glulam Beams Optimization Run Objectives") %>%
  print(include.rownames = FALSE, booktabs = TRUE)

