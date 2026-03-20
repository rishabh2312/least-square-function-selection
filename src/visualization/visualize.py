"""
Bokeh visualizations for the analysis results.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource, HoverTool, LinearColorMapper, ColorBar,
    BasicTicker, PrintfTickFormatter, Band, Title
)
from bokeh.palettes import Category10, Viridis256
from typing import Dict, List

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def _ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def _get_output_path(filename: str) -> str:
    """
    Get full path for an output file.

    Args:
        filename: Name of the file to save.

    Returns:
        Full path as string.
    """
    _ensure_output_dir()
    return str(OUTPUT_DIR / filename)


def plot_mapping_overview(train_df: pd.DataFrame,
                          ideal_df: pd.DataFrame,
                          results_df: pd.DataFrame,
                          mapped_test_df: pd.DataFrame,
                          out_html: str = "viz_overview.html"):
    """
    Create an overview plot with training, ideal and test data.

    Similar to plot_function_mapping but with simplified legend
    and different visual grouping.

    Args:
        train_df: DataFrame with training data.
        ideal_df: DataFrame with ideal functions.
        results_df: DataFrame with selection results.
        mapped_test_df: DataFrame with mapped test points.
        out_html: Output filename.
    """
    train_df = train_df.sort_values("x").reset_index(drop=True)
    ideal_df = ideal_df.sort_values("x").reset_index(drop=True)
    mapped_df = mapped_test_df.copy()

    p = figure(title="Overview: Training, Ideals, Selected Ideals, Test Points",
               width=1000, height=600, tools="pan,wheel_zoom,box_zoom,reset,save")

    for i in range(1, 51):
        cname = f"y{i}"
        if cname in ideal_df.columns:
            p.line(ideal_df["x"], ideal_df[cname], line_width=1, line_alpha=0.12, color="black")

    palette = Category10[10]
    chosen_ideals = [int(x) for x in results_df["best_ideal_fun"].tolist()]
    color_map: Dict[int, str] = {}
    for idx, ideal_no in enumerate(chosen_ideals):
        cname = f"y{ideal_no}"
        color = palette[idx % len(palette)]
        color_map[ideal_no] = color
        p.line(ideal_df["x"], ideal_df[cname], line_width=3, color=color, legend_label=f"Ideal {ideal_no}")

    training_cols = ["y1", "y2", "y3", "y4"]
    for idx, tcol in enumerate(training_cols):
        if tcol in train_df.columns:
            p.line(train_df["x"], train_df[tcol], line_width=2, line_dash="dashed",
                   color=palette[(idx + 4) % len(palette)], legend_label=f"Train {tcol}")

    assigned = mapped_df[mapped_df["ideal_function"].notna()].copy()
    unassigned = mapped_df[mapped_df["ideal_function"].isna()].copy()

    if not assigned.empty:
        for ideal_no, grp in assigned.groupby("ideal_function"):
            ideal_no = int(ideal_no)
            col = color_map.get(ideal_no, "red")
            src = ColumnDataSource(
                dict(x=grp["x"].values, y=grp["y"].values, delta=grp["delta_y"].values, ideal=[ideal_no] * len(grp)))
            glyph = p.scatter("x", "y", size=7, fill_color=col, line_color="black", source=src,
                              legend_label=f"Assigned → Ideal {ideal_no}")
            p.add_tools(HoverTool(renderers=[glyph],
                                  tooltips=[("ideal", "@ideal"), ("x", "@x"), ("y", "@y"), ("delta", "@delta")]))

    if not unassigned.empty:
        srcu = ColumnDataSource(dict(x=unassigned["x"].values, y=unassigned["y"].values))
        glyphu = p.scatter("x", "y", size=6, fill_color="lightgray", line_color="black", alpha=0.9, source=srcu,
                           legend_label="Unassigned")
        p.add_tools(HoverTool(renderers=[glyphu], tooltips=[("x", "@x"), ("y", "@y")]))

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    output_path = _get_output_path(out_html)
    output_file(output_path, title="Mapping overview")
    save(p)
    logger.info(f"Saved: {output_path}")


def plot_test_scatter_heat(mapped_test_df: pd.DataFrame, out_html: str = "viz_test_scatter.html",
                           palette=Viridis256):
    """
    Create a scatter plot of test points colored by deviation.

    Assigned points are colored based on their delta_y value using
    a color gradient. Unassigned points are shown in gray.

    Args:
        mapped_test_df: DataFrame with mapped test points.
        out_html: Output filename.
        palette: Bokeh color palette for the gradient.
    """
    df = mapped_test_df.copy()
    assigned = df[df["delta_y"].notna()].copy()
    unassigned = df[df["delta_y"].isna()].copy()

    p = figure(title="Scatter Plot with Deviation Coloring", width=900, height=400,
               tools="pan,wheel_zoom,box_zoom,reset,save")


    if not assigned.empty:
        deltas = np.array(assigned["delta_y"].values, dtype=float)
        dmin = float(deltas.min())
        dmax = float(deltas.max()) if float(deltas.max()) > float(deltas.min()) else float(deltas.min() + 1e-9)

        color_mapper = LinearColorMapper(palette=palette, low=dmin, high=dmax)

        src = ColumnDataSource(data=dict(
            x=assigned["x"].values,
            y=assigned["y"].values,
            delta=assigned["delta_y"].values
        ))

        glyph = p.scatter("x", "y", size=7,
                          fill_color={'field': 'delta', 'transform': color_mapper},
                          line_color="black", source=src)

        p.add_tools(HoverTool(renderers=[glyph], tooltips=[("x", "@x"), ("y", "@y"), ("delta", "@delta")]))

        color_bar = ColorBar(color_mapper=color_mapper,
                             ticker=BasicTicker(desired_num_ticks=8),
                             formatter=PrintfTickFormatter(format="%.4f"),
                             label_standoff=12,
                             border_line_color=None,
                             location=(0, 0))
        p.add_layout(color_bar, 'right')

    if not unassigned.empty:
        srcu = ColumnDataSource(dict(x=unassigned["x"].values, y=unassigned["y"].values))
        glyphu = p.scatter("x", "y", size=6, fill_color="lightgray", line_color="black", alpha=0.9, source=srcu)
        p.add_tools(HoverTool(renderers=[glyphu], tooltips=[("x", "@x"), ("y", "@y")]))

    output_path = _get_output_path(out_html)
    output_file(output_path, title="Test scatter heat")
    save(p)
    logger.info(f"Saved: {output_path}")


def plot_delta_histogram(mapped_test_df: pd.DataFrame, out_html: str = "viz_delta_hist.html", nbins: int = 30):
    """
    Create a histogram of deviation values.

    Shows the distribution of delta_y values for assigned test points.

    Args:
        mapped_test_df: DataFrame with mapped test points.
        out_html: Output filename.
        nbins: Number of histogram bins.
    """
    df = mapped_test_df.copy()
    assigned = df[df["delta_y"].notna()].copy()
    if assigned.empty:
        logger.warning("No assigned points to histogram.")
        return

    deltas = np.array(assigned["delta_y"].values, dtype=float)
    hist, edges = np.histogram(deltas, bins=nbins)

    p = figure(title="Histogram of delta_y (assigned test points)", width=800, height=400, tools="save")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white", fill_color="navy", alpha=0.7)
    p.xaxis.axis_label = "delta_y"
    p.yaxis.axis_label = "Count"

    src = ColumnDataSource(dict(top=hist, left=edges[:-1], right=edges[1:]))
    p.add_tools(HoverTool(tooltips=[("count", "@top"), ("range", "@left - @right")], renderers=[]))

    output_path = _get_output_path(out_html)
    output_file(output_path, title="delta histogram")
    save(p)
    logger.info(f"Saved: {output_path}")


def plot_counts_per_ideal(mapped_test_df: pd.DataFrame, out_html: str = "viz_counts_per_ideal.html"):
    """
    Create a bar chart showing assignment counts per ideal function.

    Displays how many test points were assigned to each selected
    ideal function.

    Args:
        mapped_test_df: DataFrame with mapped test points.
        out_html: Output filename.
    """
    df = mapped_test_df.copy()
    assigned = df[df["ideal_function"].notna()].copy()
    if assigned.empty:
        logger.warning("No assigned points for counts per ideal.")
        return

    counts = assigned["ideal_function"].value_counts().sort_index()
    ideals = [str(int(i)) for i in counts.index.tolist()]
    vals = counts.values.tolist()

    src = ColumnDataSource(dict(ideal=ideals, count=vals))
    p = figure(title="Assigned test points per chosen ideal", x_range=ideals,
               x_axis_label="Ideal", y_axis_label="Count", width=900, height=400, tools="save")

    bars = p.vbar(x='ideal', top='count', width=0.6, source=src, fill_color="teal", line_color="black")
    p.add_tools(HoverTool(renderers=[bars], tooltips=[("ideal", "@ideal"), ("count", "@count")]))

    p.xaxis.major_label_orientation = 1.0

    output_path = _get_output_path(out_html)
    output_file(output_path, title="counts per ideal")
    save(p)
    logger.info(f"Saved: {output_path}")


def plot_training_vs_ideal_individual(train_df: pd.DataFrame,
                                      ideal_df: pd.DataFrame,
                                      results_df: pd.DataFrame,
                                      out_html: str = "viz_training_vs_ideal.html"):
    """
    Create a 2x2 grid showing each training function vs its matched ideal.

    Each subplot shows one training function overlaid on its best-matching
    ideal function, with SSD and max deviation values displayed.

    Args:
        train_df: DataFrame with training data (x, y1-y4).
        ideal_df: DataFrame with ideal functions (x, y1-y50).
        results_df: DataFrame with selection results.
        out_html: Output filename.
    """
    train_df = train_df.sort_values("x").reset_index(drop=True)
    ideal_df = ideal_df.sort_values("x").reset_index(drop=True)

    x_train = train_df["x"].values
    x_ideal = ideal_df["x"].values

    palette = Category10[10]
    training_cols = ["y1", "y2", "y3", "y4"]
    plots: List[figure] = []

    for idx, train_col in enumerate(training_cols):
        row = results_df[results_df["training_col"] == train_col].iloc[0]
        ideal_no = int(row["best_ideal_fun"])
        ideal_col = f"y{ideal_no}"
        ssd = float(row["best_min_ssd"])
        max_dev = float(row["max_dev"])

        y_train = train_df[train_col].values
        y_ideal = ideal_df[ideal_col].values

        deviations = np.abs(y_train - y_ideal)

        p = figure(
            title=f"Training {train_col.upper()} vs Ideal Function {ideal_no}",
            width=500, height=400,
            x_axis_label="x",
            y_axis_label="y",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        ideal_src = ColumnDataSource(dict(x=x_ideal, y=y_ideal))
        ideal_line = p.line(
            "x", "y", source=ideal_src,
            line_width=2, color=palette[0],
            legend_label=f"Ideal {ideal_no}"
        )

        train_src = ColumnDataSource(dict(
            x=x_train, y=y_train,
            deviation=deviations
        ))
        train_scatter = p.scatter(
            "x", "y", source=train_src,
            size=4, color=palette[1], alpha=0.7,
            legend_label=f"Training {train_col.upper()}"
        )

        p.add_tools(HoverTool(
            renderers=[train_scatter],
            tooltips=[
                ("x", "@x{0.2f}"),
                ("y (train)", "@y{0.4f}"),
                ("deviation", "@deviation{0.4f}")
            ]
        ))

        p.add_tools(HoverTool(
            renderers=[ideal_line],
            tooltips=[
                ("x", "@x{0.2f}"),
                ("y (ideal)", "@y{0.4f}")
            ]
        ))

        p.add_layout(Title(
            text=f"SSD: {ssd:.4f} | Max Dev: {max_dev:.4f}",
            text_font_size="10pt",
            text_font_style="italic"
        ), "below")

        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "9pt"

        plots.append(p)

    grid = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]])

    output_path = _get_output_path(out_html)
    output_file(output_path, title="Training vs Ideal Functions")
    save(grid)
    logger.info(f"Saved: {output_path}")


def plot_regression_analysis(train_df: pd.DataFrame,
                            ideal_df: pd.DataFrame,
                            results_df: pd.DataFrame,
                            out_html: str = "regression_analysis.html"):
    """
    Create regression analysis showing fit quality and residuals.

    Shows how well each selected ideal function fits the training data
    with R², RMSE, and residual plots.

    Args:
        train_df: DataFrame with training data (x, y1-y4).
        ideal_df: DataFrame with ideal functions (x, y1-y50).
        results_df: DataFrame with selection results.
        out_html: Output filename.
    """
    train_df = train_df.sort_values("x").reset_index(drop=True)
    ideal_df = ideal_df.sort_values("x").reset_index(drop=True)

    palette = Category10[10]
    training_cols = ["y1", "y2", "y3", "y4"]

    plots = []

    for idx, train_col in enumerate(training_cols):
        row = results_df[results_df["training_col"] == train_col].iloc[0]
        ideal_no = int(row["best_ideal_fun"])
        ideal_col = f"y{ideal_no}"
        ssd = float(row["best_min_ssd"])

        y_train = train_df[train_col].values
        y_ideal = ideal_df[ideal_col].values
        x_vals = train_df["x"].values

        # Calculate regression metrics
        residuals = y_train - y_ideal
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_train - np.mean(y_train))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        # Residual plot
        p = figure(
            title=f"{train_col.upper()} → Ideal {ideal_no}: Residual Analysis",
            width=550, height=400,
            x_axis_label="X Values",
            y_axis_label="Residuals (Training - Ideal)",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        color = palette[idx % len(palette)]

        # Zero line
        from bokeh.models import Span
        zero_line = Span(location=0, dimension='width',
                        line_color='black', line_width=1, line_dash='dashed')
        p.add_layout(zero_line)

        # Residual scatter
        src = ColumnDataSource(dict(
            x=x_vals,
            residual=residuals,
            y_train=y_train,
            y_ideal=y_ideal
        ))

        scatter = p.scatter(
            'x', 'residual', source=src,
            size=6, color=color, alpha=0.6,
            line_color='black', line_width=0.5
        )

        p.add_tools(HoverTool(
            renderers=[scatter],
            tooltips=[
                ("X", "@x{0.2f}"),
                ("Residual", "@residual{0.4f}"),
                ("Training Y", "@y_train{0.4f}"),
                ("Ideal Y", "@y_ideal{0.4f}")
            ]
        ))

        # Add statistics annotation
        from bokeh.models import Label
        stats_text = (
            f"R² = {r_squared:.4f}\n"
            f"RMSE = {rmse:.4f}\n"
            f"MAE = {mae:.4f}\n"
            f"SSD = {ssd:.2f}"
        )

        stats_label = Label(
            x=5, y=350,
            x_units='screen', y_units='screen',
            text=stats_text,
            text_font_size="9pt",
            border_line_color="black",
            border_line_alpha=0.3,
            background_fill_color="white",
            background_fill_alpha=0.9,
            padding=5
        )
        p.add_layout(stats_label)

        plots.append(p)

    grid = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]])

    output_path = _get_output_path(out_html)
    output_file(output_path, title="Regression Analysis - Residuals")
    save(grid)
    logger.info(f"Saved: {output_path}")


def plot_training_vs_ideal_overlay(train_df: pd.DataFrame,
                                   ideal_df: pd.DataFrame,
                                   results_df: pd.DataFrame,
                                   out_html: str = "training_vs_ideal_overlay.html"):
    """
    Create a single overlay plot showing all training functions and their selected ideals.

    This plot clearly shows how each training function matches its ideal function
    by displaying them together with distinct colors and clear legends.

    Args:
        train_df: DataFrame with training data (x, y1-y4).
        ideal_df: DataFrame with ideal functions (x, y1-y50).
        results_df: DataFrame with selection results.
        out_html: Output filename.
    """
    train_df = train_df.sort_values("x").reset_index(drop=True)
    ideal_df = ideal_df.sort_values("x").reset_index(drop=True)

    x_train = train_df["x"].values
    x_ideal = ideal_df["x"].values

    palette = Category10[10]
    training_cols = ["y1", "y2", "y3", "y4"]

    p = figure(
        title="Training Functions vs Selected Ideal Functions - Overlay Comparison",
        width=1200, height=600,
        x_axis_label="x",
        y_axis_label="y",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    for idx, train_col in enumerate(training_cols):
        row = results_df[results_df["training_col"] == train_col].iloc[0]
        ideal_no = int(row["best_ideal_fun"])
        ideal_col = f"y{ideal_no}"

        y_train = train_df[train_col].values
        y_ideal = ideal_df[ideal_col].values

        color = palette[idx]

        ideal_src = ColumnDataSource(dict(x=x_ideal, y=y_ideal))
        ideal_line = p.line(
            "x", "y", source=ideal_src,
            line_width=3, color=color, alpha=0.8,
            legend_label=f"Ideal {ideal_no} (for {train_col.upper()})"
        )

        train_src = ColumnDataSource(dict(x=x_train, y=y_train))
        train_scatter = p.scatter(
            "x", "y", source=train_src,
            size=6, color=color, alpha=0.5, marker="circle",
            legend_label=f"Training {train_col.upper()}"
        )

        p.add_tools(HoverTool(
            renderers=[ideal_line],
            tooltips=[
                ("Type", f"Ideal {ideal_no}"),
                ("x", "@x{0.2f}"),
                ("y", "@y{0.4f}")
            ]
        ))

        p.add_tools(HoverTool(
            renderers=[train_scatter],
            tooltips=[
                ("Type", f"Training {train_col.upper()}"),
                ("x", "@x{0.2f}"),
                ("y", "@y{0.4f}")
            ]
        ))

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "10pt"
    p.legend.background_fill_alpha = 0.8

    from bokeh.models import Title
    subtitle = Title(
        text="Lines: Selected Ideal Functions | Points: Training Data",
        text_font_size="11pt",
        text_font_style="italic"
    )
    p.add_layout(subtitle, "above")

    output_path = _get_output_path(out_html)
    output_file(output_path, title="Training vs Ideal Overlay")
    save(p)
    logger.info(f"Saved: {output_path}")


def plot_assigned_vs_unassigned(mapped_test_df: pd.DataFrame,
                                ideal_df: pd.DataFrame,
                                results_df: pd.DataFrame,
                                out_html: str = "assigned_vs_unassigned.html"):
    """
    Create a clear visualization showing assigned vs unassigned test points.

    This plot distinctly shows which test points were successfully mapped to
    ideal functions and which ones couldn't be assigned, with clear visual
    separation and statistics.

    Args:
        mapped_test_df: DataFrame with mapped test points.
        ideal_df: DataFrame with ideal functions.
        results_df: DataFrame with selection results.
        out_html: Output filename.
    """
    df = mapped_test_df.copy()
    ideal_df = ideal_df.sort_values("x").reset_index(drop=True)

    assigned = df[df["ideal_function"].notna()].copy()
    unassigned = df[df["ideal_function"].isna()].copy()

    total_points = len(df)
    assigned_count = len(assigned)
    unassigned_count = len(unassigned)

    p = figure(
        title=f"Test Points Assignment Status - {assigned_count} Assigned, {unassigned_count} Unassigned (Total: {total_points})",
        width=1200, height=600,
        x_axis_label="x",
        y_axis_label="y",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    chosen_ideals = [int(x) for x in results_df["best_ideal_fun"].tolist()]
    palette = Category10[10]

    for idx, ideal_no in enumerate(chosen_ideals):
        cname = f"y{ideal_no}"
        if cname in ideal_df.columns:
            color = palette[idx % len(palette)]
            p.line(ideal_df["x"], ideal_df[cname],
                   line_width=2, color=color, alpha=0.3,
                   legend_label=f"Ideal {ideal_no}")

    if not assigned.empty:
        for ideal_no, grp in assigned.groupby("ideal_function"):
            ideal_no = int(ideal_no)
            idx = chosen_ideals.index(ideal_no) if ideal_no in chosen_ideals else 0
            col = palette[idx % len(palette)]

            src = ColumnDataSource(dict(
                x=grp["x"].values,
                y=grp["y"].values,
                delta=grp["delta_y"].values,
                ideal=[ideal_no] * len(grp)
            ))

            glyph = p.scatter(
                "x", "y", size=10, fill_color=col,
                line_color="darkgreen", line_width=2,
                source=src, marker="circle",
                legend_label=f"Assigned to Ideal {ideal_no}"
            )

            p.add_tools(HoverTool(
                renderers=[glyph],
                tooltips=[
                    ("Status", "ASSIGNED"),
                    ("Ideal Function", "@ideal"),
                    ("x", "@x{0.2f}"),
                    ("y", "@y{0.4f}"),
                    ("Deviation", "@delta{0.4f}")
                ]
            ))

    if not unassigned.empty:
        srcu = ColumnDataSource(dict(
            x=unassigned["x"].values,
            y=unassigned["y"].values
        ))

        glyphu = p.scatter(
            "x", "y", size=12,
            fill_color="red", line_color="darkred",
            line_width=2, alpha=0.8, source=srcu,
            marker="x",
            legend_label=f"Unassigned ({unassigned_count} points)"
        )

        p.add_tools(HoverTool(
            renderers=[glyphu],
            tooltips=[
                ("Status", "UNASSIGNED"),
                ("x", "@x{0.2f}"),
                ("y", "@y{0.4f}"),
                ("Reason", "Exceeds threshold")
            ]
        ))

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "10pt"
    p.legend.background_fill_alpha = 0.9

    from bokeh.models import Title
    subtitle = Title(
        text="Green circles: Successfully assigned | Red X: Failed assignment (deviation too large)",
        text_font_size="11pt",
        text_font_style="italic",
        text_color="navy"
    )
    p.add_layout(subtitle, "above")

    output_path = _get_output_path(out_html)
    output_file(output_path, title="Assigned vs Unassigned")
    save(p)
    logger.info(f"Saved: {output_path}")
