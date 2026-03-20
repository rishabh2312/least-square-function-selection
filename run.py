"""
Main script - runs the full analysis pipeline and generates visualizations.
"""

from pathlib import Path
from src.database.data_operations import publish_csvs_to_db, read_data, persist_mapped_test_results_update
from src.service.ideal_function_selector import least_squares_select, map_test_points_from_results, summary_print
from src.database.table_creation import Training, IdealFunction, TestResult
from src.visualization.visualize import (
    plot_mapping_overview, plot_test_scatter_heat,
    plot_training_vs_ideal_individual,
    plot_regression_analysis, plot_training_vs_ideal_overlay,
    plot_assigned_vs_unassigned
)
from src.loader.csv_reader import CSVReader

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    publish_csvs_to_db(BASE_DIR)

    training_data = read_data(Training)
    ideal_data = read_data(IdealFunction)
    test_data = read_data(TestResult)

    results_df = least_squares_select(training_data, ideal_data)
    mapped_result_df = map_test_points_from_results(test_data, ideal_data, results_df)
    persist_mapped_test_results_update(mapped_result_df)
    summary_print(mapped_result_df)

    loader = CSVReader(BASE_DIR)

    training_df = loader.load_csv_data("train.csv")
    idea_df = loader.load_csv_data("ideal.csv")
    test_results_df = loader.load_csv_data("test.csv")

    plot_mapping_overview(training_df, idea_df, results_df, mapped_result_df,
                          out_html="1_complete_overview.html")

    plot_training_vs_ideal_individual(training_df, idea_df, results_df,
                                      out_html="2_training_vs_ideal_grid.html")

    plot_test_scatter_heat(mapped_result_df,
                           out_html="3_test_points_deviation_heatmap.html")

    plot_regression_analysis(training_df, idea_df, results_df,
                             out_html="4_regression_analysis.html")

    plot_training_vs_ideal_overlay(training_df, idea_df, results_df,
                                   out_html="5_training_vs_ideal_overlay.html")

    plot_assigned_vs_unassigned(mapped_result_df, idea_df, results_df,
                                out_html="6_assigned_vs_unassigned_points.html")
