# Customer-level cross-validation --------------------------------------------

mean_cv_accuracy <- function(data, config, fit_function, parameters, seed) {
  folds <- make_customer_folds(
    data[[config$customer_id]],
    n_folds = config$cv_folds,
    seed = seed
  )

  scores <- vapply(seq_along(folds), function(fold_index) {
    validation_ids <- folds[[fold_index]]
    fold_train <- data[!data[[config$customer_id]] %in% validation_ids, , drop = FALSE]
    fold_valid <- data[data[[config$customer_id]] %in% validation_ids, , drop = FALSE]

    result <- do.call(fit_function, c(list(
      train = fold_train,
      test = fold_valid,
      target = config$target,
      customer_id = config$customer_id
    ), parameters))

    classification_metrics(
      actual = fold_valid[[config$target]],
      predicted = result$predicted,
      class_labels = config$class_labels,
      optimistic_error_weight = config$optimistic_error_weight
    )$accuracy
  }, numeric(1))

  mean(scores, na.rm = TRUE)
}

tune_model <- function(data, config, grid, fit_function, seed) {
  scores <- vapply(seq_len(nrow(grid)), function(i) {
    message("  Parameter set ", i, " of ", nrow(grid))
    mean_cv_accuracy(
      data = data,
      config = config,
      fit_function = fit_function,
      parameters = as.list(grid[i, , drop = FALSE]),
      seed = seed
    )
  }, numeric(1))

  results <- cbind(grid, cv_accuracy = scores)
  results[which.max(results$cv_accuracy), , drop = FALSE]
}

make_ensemble_weights <- function(n_models, step = 0.25) {
  candidates <- seq(0, 1, by = step)
  grid <- expand.grid(rep(list(candidates), n_models))
  grid[abs(rowSums(grid) - 1) < 1e-9, , drop = FALSE]
}

