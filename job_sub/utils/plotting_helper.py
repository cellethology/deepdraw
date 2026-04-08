"""Plotting helpers shared across analysis notebooks."""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_STRATEGY_COLORS = {
    "Deepdraw": "#1f77b4",
    "onehot": "#ff7f0e",
    "equal": "#d0d0d0",
    "random": "#2ca02c",
}


def _normalize_task_label(task: object) -> str:
    """Convert grouped labels like (SA, SA) to a single display label."""
    if isinstance(task, tuple) and len(task) > 0:
        return str(task[0])
    return str(task)


def plot_prop_for_col(
    df,
    col: str,
    order: Sequence[str],
    ax,
    task_order: Sequence[str] | None = None,
    color_map: dict[str, str] | None = None,
    show_legend: bool = True,
    legend_title: str = "Best strategy",
    legend_kwargs: dict | None = None,
) -> None:
    """Plot stacked strategy proportions by dataset_group for one comparison column."""
    colors = color_map or DEFAULT_STRATEGY_COLORS
    plot_df = df.copy()
    plot_df = plot_df[plot_df[col].notna()]  # drop missing comparisons
    plot_df["best_strategy"] = plot_df[col].astype(str).fillna(plot_df[col])

    prop_table = (
        plot_df.groupby(["dataset_group", "best_strategy"])
        .size()
        .groupby(level=0)
        .apply(lambda counts: counts / counts.sum())
        .unstack(fill_value=0)
    )
    prop_table = prop_table.reindex(columns=order, fill_value=0)
    prop_table.index = [_normalize_task_label(task) for task in prop_table.index]
    prop_table = prop_table.groupby(level=0, sort=False).sum()

    if task_order is not None:
        ordered_tasks = [task for task in task_order if task in prop_table.index]
        remaining_tasks = [task for task in prop_table.index if task not in task_order]
        prop_table = prop_table.reindex(ordered_tasks + remaining_tasks)

    print(prop_table)

    prop_table.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[colors.get(c, "#333333") for c in prop_table.columns],
        legend=False,
    )
    ax.set_ylabel("Proportion of \n design spaces", fontsize=11)
    ax.set_xlabel("Design task", fontsize=11)
    ax.set_xticks(range(len(prop_table.index)))
    ax.set_xticklabels([str(label) for label in prop_table.index], rotation=0)

    if show_legend:
        kwargs = {
            "title": legend_title,
            "bbox_to_anchor": (1.05, 1),
            "loc": "upper left",
            "frameon": False,
        }
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
