library(tidyverse)
library(cowplot)
theme_set(theme_minimal(base_size = 10)) # Adjust the base_size as needed

read_json <- function(file_name) {
  json <- jsonlite::fromJSON(file_name)
  depth <- json$depth
  generations <- 1:length(json$stats$gen)
  run <- gsub(paste0('soln_ES_d', depth, '_(\\d+).json[.part]*'), '\\1', basename(file_name))
  print(c(file_name, run))
  stats_tibble <- tibble(
    generation = json$stats$gen,
    opt_waste = json$stats$waste,
    opt_presses = json$stats$npresses,
    depth = depth,
    run = as.factor(run),
  ) %>%
    mutate(
      x = map(generations, ~json$stats$x[[.]]),
      xstar = map(generations, ~json$stats$xstar[[.]])
    ) %>%
    cbind(json$stats$run_summary)
  return(stats_tibble)
}

read_log <- function(file_name) {
  # read the log file it is a csv with a header
  log <- read.csv(file_name, header = T)
  # get the depth from the file name
  depth <- as.integer(gsub(paste0('ES_', '(\\d+)_.*'), '\\1', basename(file_name)))
  # get the run number from the file name
  run <- as.integer(gsub(paste0('ES_\\d+_(\\d+)_.*'), '\\1', basename(file_name)))
  log %>% mutate(depth = depth, run = run)
}


area_press <- 25000 * 45 * 26 / 1e6