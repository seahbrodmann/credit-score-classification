# Evaluation metrics ---------------------------------------------------------

make_confusion_matrix <- function(actual, predicted, class_labels) {
  actual <- factor(actual, levels = class_labels)
  predicted <- factor(predicted, levels = class_labels)
  table(actual = actual, predicted = predicted)
}

safe_divide <- function(numerator, denominator) {
  ifelse(denominator == 0, NA_real_, numerator / denominator)
}

classification_metrics <- function(
    actual,
    predicted,
    class_labels = c("Bad", "Standard", "Good"),
    optimistic_error_weight = 5) {

  cm <- make_confusion_matrix(actual, predicted, class_labels)
  total <- sum(cm)
  correct <- sum(diag(cm))

  recall_by_class <- safe_divide(diag(cm), rowSums(cm))
  specificity_by_class <- vapply(seq_along(class_labels), function(i) {
    true_negative <- sum(cm[-i, -i, drop = FALSE])
    negative_total <- sum(cm[-i, , drop = FALSE])
    safe_divide(true_negative, negative_total)
  }, numeric(1))

  # A Bad customer classified as Good is treated as five errors. This retains
  # the business-risk assumption used in the original project.
  weighted_total <- total +
    (optimistic_error_weight - 1) * cm["Bad", "Good"]
  cost_weighted_accuracy <- safe_divide(correct, weighted_total)

  off_diagonal_errors <- total - correct
  optimistic_errors <-
    cm["Bad", "Standard"] + cm["Bad", "Good"] + cm["Standard", "Good"]

  data.frame(
    accuracy = safe_divide(correct, total),
    macro_recall = mean(recall_by_class, na.rm = TRUE),
    macro_specificity = mean(specificity_by_class, na.rm = TRUE),
    cost_weighted_accuracy = cost_weighted_accuracy,
    optimistic_error_ratio = safe_divide(optimistic_errors, off_diagonal_errors)
  )
}

evaluate_predictions <- function(actual, predicted, model_name, config) {
  metrics <- classification_metrics(
    actual = actual,
    predicted = predicted,
    class_labels = config$class_labels,
    optimistic_error_weight = config$optimistic_error_weight
  )
  metrics$model <- model_name
  metrics[, c("model", setdiff(names(metrics), "model"))]
}

