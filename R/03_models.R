# Model training and prediction ----------------------------------------------

prepare_model_matrix <- function(train, test, target, customer_id) {
  train_model <- train[, setdiff(names(train), customer_id), drop = FALSE]
  test_model <- test[, setdiff(names(test), customer_id), drop = FALSE]

  formula <- stats::as.formula(paste(target, "~ ."))
  train_x <- stats::model.matrix(formula, train_model)[, -1, drop = FALSE]
  test_x <- stats::model.matrix(formula, test_model)[, -1, drop = FALSE]

  missing_in_test <- setdiff(colnames(train_x), colnames(test_x))
  for (column in missing_in_test) test_x <- cbind(test_x, setNames(data.frame(0), column))
  extra_in_test <- setdiff(colnames(test_x), colnames(train_x))
  if (length(extra_in_test) > 0) test_x <- test_x[, !colnames(test_x) %in% extra_in_test, drop = FALSE]
  test_x <- test_x[, colnames(train_x), drop = FALSE]

  list(
    train_x = train_x,
    test_x = as.matrix(test_x),
    train_y = train_model[[target]],
    test_y = test_model[[target]]
  )
}

fit_elastic_net <- function(train, test, target, customer_id, alpha, lambda) {
  matrices <- prepare_model_matrix(train, test, target, customer_id)
  fit <- glmnet::glmnet(
    x = matrices$train_x,
    y = matrices$train_y,
    family = "multinomial",
    alpha = alpha,
    lambda = lambda
  )
  probabilities <- predict(fit, matrices$test_x, type = "response")[, , 1]
  predicted <- colnames(probabilities)[max.col(probabilities, ties.method = "first")]
  list(model = fit, predicted = predicted, probabilities = probabilities)
}

fit_random_forest <- function(train, test, target, customer_id, ntree, mtry, nodesize) {
  train_model <- train[, setdiff(names(train), customer_id), drop = FALSE]
  test_model <- test[, setdiff(names(test), customer_id), drop = FALSE]
  predictors <- setdiff(names(train_model), target)
  mtry <- min(as.integer(mtry), length(predictors))

  fit <- randomForest::randomForest(
    stats::reformulate(predictors, response = target),
    data = train_model,
    ntree = as.integer(ntree),
    mtry = mtry,
    nodesize = as.integer(nodesize)
  )
  probabilities <- predict(fit, test_model, type = "prob")
  predicted <- colnames(probabilities)[max.col(probabilities, ties.method = "first")]
  list(model = fit, predicted = predicted, probabilities = probabilities)
}

fit_xgboost <- function(train, test, target, customer_id, eta, max_depth, nrounds) {
  matrices <- prepare_model_matrix(train, test, target, customer_id)
  class_levels <- levels(matrices$train_y)
  train_y <- as.integer(matrices$train_y) - 1L

  fit <- xgboost::xgboost(
    data = xgboost::xgb.DMatrix(matrices$train_x, label = train_y),
    objective = "multi:softprob",
    num_class = length(class_levels),
    eta = eta,
    max_depth = as.integer(max_depth),
    nrounds = as.integer(nrounds),
    verbose = 0
  )

  raw_probabilities <- predict(fit, matrices$test_x)
  probabilities <- matrix(
    raw_probabilities,
    ncol = length(class_levels),
    byrow = TRUE,
    dimnames = list(NULL, class_levels)
  )
  predicted <- class_levels[max.col(probabilities, ties.method = "first")]
  list(model = fit, predicted = predicted, probabilities = probabilities)
}

fit_lightgbm <- function(train, test, target, customer_id, params, seed) {
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("LightGBM is enabled but the lightgbm package is not installed.", call. = FALSE)
  }

  matrices <- prepare_model_matrix(train, test, target, customer_id)
  class_levels <- levels(matrices$train_y)
  labels <- as.integer(matrices$train_y) - 1L

  set.seed(seed)
  fit <- lightgbm::lgb.train(
    params = list(
      objective = "multiclass",
      metric = "multi_error",
      num_class = length(class_levels),
      learning_rate = params$learning_rate,
      max_depth = as.integer(params$max_depth),
      seed = as.integer(seed),
      verbosity = -1
    ),
    data = lightgbm::lgb.Dataset(as.matrix(matrices$train_x), label = labels),
    nrounds = as.integer(params$num_iterations),
    verbose = -1
  )

  raw_probabilities <- predict(fit, as.matrix(matrices$test_x))
  probabilities <- matrix(
    raw_probabilities,
    ncol = length(class_levels),
    byrow = TRUE,
    dimnames = list(NULL, class_levels)
  )
  predicted <- class_levels[max.col(probabilities, ties.method = "first")]
  list(model = fit, predicted = predicted, probabilities = probabilities)
}

weighted_ensemble <- function(probability_list, weights) {
  if (length(probability_list) != length(weights)) {
    stop("Provide one ensemble weight for each probability matrix.", call. = FALSE)
  }
  weights <- weights / sum(weights)
  combined <- probability_list[[1]] * weights[1]
  if (length(probability_list) > 1) {
    for (i in 2:length(probability_list)) {
      combined <- combined + probability_list[[i]] * weights[i]
    }
  }
  colnames(combined)[max.col(combined, ties.method = "first")]
}

