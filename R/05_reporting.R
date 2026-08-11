# Result tables and figures --------------------------------------------------

save_model_comparison <- function(results, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(results, file.path(output_dir, "model_comparison.csv"))
}

plot_model_comparison <- function(results, output_dir) {
  figure_dir <- file.path(output_dir, "figures")
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

  plot_data <- results |>
    dplyr::group_by(model) |>
    dplyr::summarise(
      accuracy = mean(accuracy, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::arrange(accuracy)

  chart <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = stats::reorder(model, accuracy), y = accuracy)
  ) +
    ggplot2::geom_col(fill = "#3D8DFF", width = 0.7) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(labels = scales::label_percent(accuracy = 0.1)) +
    ggplot2::labs(
      title = "Model accuracy comparison",
      x = NULL,
      y = "Mean test accuracy"
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())

  ggplot2::ggsave(
    file.path(figure_dir, "model_accuracy.png"),
    chart,
    width = 8,
    height = 5,
    dpi = 200
  )
}

