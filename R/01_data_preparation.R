# Data loading and validation -------------------------------------------------

required_packages <- c(
  "dplyr", "tidyr", "readr", "ggplot2", "scales", "glmnet",
  "randomForest", "xgboost"
)

check_packages <- function(packages = required_packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Install the following packages before running the project: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
}

load_credit_data <- function(path, config) {
  if (!file.exists(path)) {
    stop(
      "Data file not found at '", path, "'. See data/README.md for the expected schema.",
      call. = FALSE
    )
  }

  data <- readr::read_csv(path, show_col_types = FALSE)
  required_columns <- c(
    config$customer_id,
    config$target,
    config$previous_score,
    config$month
  )
  missing_columns <- setdiff(required_columns, names(data))

  if (length(missing_columns) > 0) {
    stop(
      "Missing required columns: ", paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  if (anyDuplicated(data[c(config$customer_id, config$month)]) > 0) {
    stop("Customer_ID and Month must uniquely identify each row.", call. = FALSE)
  }

  data[[config$target]] <- factor(
    data[[config$target]],
    levels = config$class_levels,
    labels = config$class_labels
  )
  data[[config$previous_score]] <- factor(
    data[[config$previous_score]],
    levels = config$class_levels,
    labels = config$class_labels
  )

  if ("Occupation" %in% names(data)) {
    data$Occupation <- factor(data$Occupation)
  }

  # The previous-month score is unavailable in month 1. The original one-step
  # experiment therefore starts in month 2.
  data <- dplyr::filter(data, .data[[config$month]] != 1)

  if (anyNA(data[[config$target]])) {
    stop("Credit_Score contains missing or unexpected class values.", call. = FALSE)
  }

  data
}

make_customer_folds <- function(customer_ids, n_folds = 5L, seed = 2026L) {
  unique_ids <- unique(customer_ids)
  if (length(unique_ids) < n_folds) {
    stop("The number of customers must be at least the number of folds.", call. = FALSE)
  }

  set.seed(seed)
  shuffled_ids <- sample(unique_ids)
  split(shuffled_ids, rep(seq_len(n_folds), length.out = length(shuffled_ids)))
}

split_by_customer <- function(data, customer_id, test_customers, seed) {
  customer_ids <- unique(data[[customer_id]])
  n_test <- min(as.integer(test_customers), length(customer_ids) - 1L)
  if (n_test < 1L) stop("At least two customers are required.", call. = FALSE)

  set.seed(seed)
  test_ids <- sample(customer_ids, n_test)

  list(
    train = data[!data[[customer_id]] %in% test_ids, , drop = FALSE],
    test = data[data[[customer_id]] %in% test_ids, , drop = FALSE]
  )
}

sample_one_month_per_customer <- function(data, customer_id, seed) {
  set.seed(seed)
  data |>
    dplyr::group_by(.data[[customer_id]]) |>
    dplyr::slice_sample(n = 1) |>
    dplyr::ungroup()
}

drop_non_model_columns <- function(data) {
  removable <- intersect(c("X"), names(data))
  dplyr::select(data, -dplyr::all_of(removable))
}
