from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

plotting_helper = pytest.importorskip(
    "job_sub.utils.analysis.plotting_helper",
    reason="analysis/ folder is gitignored; plotting_helper not present on this checkout",
)
plot_prop_for_col = plotting_helper.plot_prop_for_col


def test_plot_prop_for_col_scales_to_percent_and_sets_titles() -> None:
    df = pd.DataFrame(
        {
            "dataset_group": [("Task A", "Task A"), ("Task A", "Task A"), ("Task B",)],
            "winner": ["one-hot", "random", "one-hot"],
        }
    )

    fig, ax = plt.subplots()
    try:
        plot_prop_for_col(
            df,
            col="winner",
            order=["one-hot", "random", "equal"],
            ax=ax,
            task_order=["Task B", "Task A"],
            show_legend=False,
            scale_to_percent=True,
            title="Strategy proportions",
            subtitle="Normalized by task",
        )

        assert ax.get_ylabel() == "% of design spaces"
        assert ax.get_ylim() == (0.0, 100.0)
        assert ax.get_title() == "Strategy proportions"
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "Task B",
            "Task A",
        ]
        assert any(text.get_text() == "Normalized by task" for text in ax.texts)
        assert len(ax.patches) == 6
    finally:
        plt.close(fig)


def test_plot_prop_for_col_handles_empty_input() -> None:
    df = pd.DataFrame({"dataset_group": ["Task A"], "winner": [None]})

    fig, ax = plt.subplots()
    try:
        plot_prop_for_col(
            df,
            col="winner",
            order=["one-hot", "random"],
            ax=ax,
            scale_to_percent=True,
            title="Empty plot",
            subtitle="No qualifying rows",
        )

        assert ax.get_xlabel() == "Design task"
        assert ax.get_ylabel() == "% of design spaces"
        assert ax.get_ylim() == (0.0, 100.0)
        assert ax.get_title() == "Empty plot"
        assert any(text.get_text() == "No data" for text in ax.texts)
        assert any(text.get_text() == "No qualifying rows" for text in ax.texts)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
    finally:
        plt.close(fig)
