#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(did)
  library(broom)
})

set.seed(20240924)

data_path <- Sys.getenv(
  "JEL_DID_DATA",
  "/Users/gabrielsaco/Documents/GitHub/JEL-DiD/data/county_mortality_data.csv"
)

out_dir <- file.path("report", "artifacts")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

covs <- c(
  "perc_female", "perc_white", "perc_hispanic",
  "unemp_rate", "poverty_rate", "median_income"
)
boot_iters <- 25000

# Load and clean replication data (mirrors Code_Appendix.qmd Table 7 steps)
mydata <- read_csv(data_path, show_col_types = FALSE) |>
  mutate(state = str_sub(county, nchar(county) - 1, nchar(county))) |>
  filter(!(state %in% c("DC", "DE", "MA", "NY", "VT"))) |>
  filter(yaca == 2014 | is.na(yaca) | yaca > 2019) |>
  mutate(
    perc_white = population_20_64_white / population_20_64 * 100,
    perc_hispanic = population_20_64_hispanic / population_20_64 * 100,
    perc_female = population_20_64_female / population_20_64 * 100,
    unemp_rate = unemp_rate * 100,
    median_income = median_income / 1000
  ) |>
  select(
    state, county, county_code, year, population_20_64, yaca,
    starts_with("perc_"), crude_rate_20_64, all_of(covs)
  )

# Drop missing values in all columns except yaca
cols_to_check <- setdiff(names(mydata), "yaca")
mydata <- tidyr::drop_na(mydata, all_of(cols_to_check))

mydata <- mydata |>
  group_by(county_code) |>
  filter(length(which(year %in% c(2013, 2014))) == 2) |>
  ungroup()

mydata <- mydata |>
  group_by(county_code) |>
  drop_na(crude_rate_20_64) |>
  filter(n() == 11) |>
  ungroup()

short_data <- mydata |>
  mutate(
    Treat = if_else(yaca == 2014 & !is.na(yaca), 1, 0),
    Post = if_else(year == 2014, 1, 0)
  ) |>
  filter(year %in% c(2013, 2014)) |>
  group_by(county_code) |>
  mutate(set_wt = population_20_64[year == 2013][1]) |>
  ungroup()

data_cs <- short_data |>
  mutate(
    treat_year = if_else(yaca == 2014 & !is.na(yaca), 2014, 0),
    county_code = as.numeric(county_code)
  )

run_cs <- function(method, wt_col) {
  atts <- att_gt(
    yname = "crude_rate_20_64",
    tname = "year",
    idname = "county_code",
    gname = "treat_year",
    xformla = as.formula(paste("~", paste(covs, collapse = "+"))),
    data = data_cs,
    panel = TRUE,
    control_group = "nevertreated",
    bstrap = TRUE,
    cband = TRUE,
    est_method = method,
    weightsname = wt_col,
    base_period = "universal",
    biters = boot_iters,
    print_details = FALSE
  )

  aggte(atts, na.rm = TRUE, biters = boot_iters) |>
    tidy() |>
    filter(group == 2014) |>
    transmute(
      spec_id = "medicaid_table7",
      method = method,
      weighted = !is.null(wt_col),
      agg_type = "group",
      g = group,
      t = group,
      e = 0,
      estimate = estimate,
      se = std.error,
      ci_lo = conf.low,
      ci_hi = conf.high
    )
}

results <- bind_rows(
  run_cs("reg", NULL),
  run_cs("ipw", NULL),
  run_cs("dr", NULL),
  run_cs("reg", "set_wt"),
  run_cs("ipw", "set_wt"),
  run_cs("dr", "set_wt")
)

write_csv(results, file.path(out_dir, "truth_r.csv"))

meta <- tibble(
  source = "truth_r",
  n_rows = nrow(data_cs),
  n_units = n_distinct(data_cs$county_code),
  treated_units = n_distinct(data_cs$county_code[data_cs$treat_year == 2014]),
  control_units = n_distinct(data_cs$county_code[data_cs$treat_year == 0]),
  weight_min = min(data_cs$set_wt, na.rm = TRUE),
  weight_mean = mean(data_cs$set_wt, na.rm = TRUE),
  weight_max = max(data_cs$set_wt, na.rm = TRUE)
)

write_csv(meta, file.path(out_dir, "meta_r.csv"))

sink(file.path(out_dir, "session_info_r.txt"))
sessionInfo()
sink()
