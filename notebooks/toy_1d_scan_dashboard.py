"""
Interactive dashboard for visualizing the toy_1d_scan sweep results.

Run with:
    panel serve notebooks/toy_1d_scan_dashboard.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.colors import qualitative

pn.extension(
    "plotly",
    sizing_mode="stretch_width",
    raw_css=[
        ".svd-row {border: 1px solid #d9d9d9; border-radius: 8px; padding: 8px;}",
        ".svd-row.selected-row {border-color: #1f77b4; box-shadow: 0 0 0 2px rgba(31,119,180,0.2);}",
    ],
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = (
    REPO_ROOT / "experiment_results" / "toy_1d_scan" / "toy_1d_scan_results_df.pkl"
)


@dataclass(frozen=True)
class ComparisonConfig:
    optimizer: str
    lr: float


@dataclass(frozen=True)
class RowConfig:
    k_fraction: float
    lr: float
    batch_sizes: Tuple[int, ...]
    comparisons: Tuple[ComparisonConfig, ...]
    smoothing_window: int
    loss_curve: str
    epoch_batch_scale: str
    raw_batch_scale: str
    rank_scale: str
    epoch_batch_x_range: Tuple[float, float]
    epoch_batch_y_range: Tuple[float, float]
    raw_batch_x_range: Tuple[float, float]
    raw_batch_y_range: Tuple[float, float]
    rank_x_range: Tuple[float, float]
    rank_y_range: Tuple[float, float]


class Toy1DScanDashboard:
    """Panel app that lets you slice and plot the toy_1d_scan outputs."""

    def __init__(self, results_path: Path = DEFAULT_RESULTS):
        self.results_path = Path(results_path)
        if not self.results_path.exists():
            raise FileNotFoundError(
                f"Could not find scan results at {self.results_path}. "
                "Run the sweep first or update `DEFAULT_RESULTS`."
            )

        self.df = pd.read_pickle(self.results_path)
        self.svd_df = self.df[self.df["optimizer"] == "SVD"].copy()
        self.standard_df = self.df[self.df["optimizer"] != "SVD"].copy()
        if self.svd_df.empty:
            raise ValueError("No SVD runs found in the results file.")

        self.batch_sizes = tuple(
            int(bs) for bs in sorted(self.svd_df["batch_size"].unique())
        )
        self.k_fractions = tuple(
            float(kf) for kf in sorted(self.svd_df["k_fraction"].unique())
        )
        self.svd_lrs = tuple(float(lr) for lr in sorted(self.svd_df["lr"].unique()))
        self.compare_options = self._build_comparison_options()

        self.batch_colors = self._build_color_map(
            keys=self.batch_sizes, palette=qualitative.Bold
        )
        self.compare_colors = self._build_color_map(
            keys=self.compare_options.values(), palette=qualitative.Safe
        )

        self.max_epochs = self._compute_max_len("train")
        self.max_batches = self._compute_max_len("train_batch")
        self.loss_min, self.loss_max = self._compute_loss_bounds()
        self.rank_max = self._compute_rank_max()

        self.row_configs: Dict[pn.Column, RowConfig] = {}
        self.selected_row: pn.Column | None = None

        self._init_widgets()
        self._init_layout()

    @property
    def panel(self) -> pn.Template:
        """Expose the Panel template so panel serve can pick it up."""
        return self.template

    # ------------------------------------------------------------------ UI init
    def _init_widgets(self) -> None:
        self.lr_slider = pn.widgets.DiscreteSlider(
            name="SVD learning rate",
            options=self._format_slider_options(self.svd_lrs),
            value=self.svd_lrs[1] if len(self.svd_lrs) > 1 else self.svd_lrs[0],
        )
        self.k_slider = pn.widgets.DiscreteSlider(
            name="k fraction",
            options=self._format_slider_options(self.k_fractions),
            value=self.k_fractions[2] if len(self.k_fractions) > 2 else self.k_fractions[0],
        )
        self.batch_selector = pn.widgets.MultiChoice(
            name="Batch sizes",
            options=list(self.batch_sizes),
            value=list(self.batch_sizes),
        )
        self.comparison_selector = pn.widgets.MultiChoice(
            name="Overlay optimizers (train/val)",
            options=self.compare_options,
            value=[],
        )
        max_history = max(
            len(loss.get("train_batch", []))
            for loss in self.svd_df["losses"].tolist()
        )
        self.svd_window = pn.widgets.IntSlider(
            name="SVD rank smoothing window",
            start=1,
            end=max(5, min(200, max_history)),
            value=10,
            step=1,
        )
        self.loss_curve_toggle = pn.widgets.RadioButtonGroup(
            name="Loss curve",
            options=["train", "val"],
            button_type="primary",
            value="train",
        )
        self.epoch_batch_scale_toggle = pn.widgets.RadioButtonGroup(
            name="Epoch-scaled loss scale",
            options=["log", "linear"],
            button_type="primary",
            value="log",
        )
        self.raw_batch_scale_toggle = pn.widgets.RadioButtonGroup(
            name="Raw batch loss scale",
            options=["log", "linear"],
            button_type="primary",
            value="log",
        )
        self.rank_scale_toggle = pn.widgets.RadioButtonGroup(
            name="Rank plot scale",
            options=["log", "linear"],
            button_type="primary",
            value="linear",
        )
        def _range_box(name, start, end):
            return pn.widgets.TextInput(
                name=name,
                value=f"{start:.3g},{end:.3g}",
                placeholder="min,max",
            )

        self.epoch_batch_x_range = _range_box(
            "Epoch-scaled x-range", 0.0, float(self.max_epochs)
        )
        self.epoch_batch_y_range = _range_box(
            "Epoch-scaled y-range", float(self.loss_min), float(self.loss_max)
        )
        self.raw_batch_x_range = _range_box(
            "Raw batch x-range", 0.0, float(self.max_batches)
        )
        self.raw_batch_y_range = _range_box(
            "Raw batch y-range", float(self.loss_min), float(self.loss_max)
        )
        self.rank_x_range = _range_box(
            "Rank x-range", 0.0, float(self.max_batches)
        )
        self.rank_y_range = _range_box(
            "Rank y-range", 0.0, float(self.rank_max)
        )
        self.sync_axes_btn = pn.widgets.Button(
            name="Apply axis settings to all rows",
            button_type="primary",
            icon="arrows-expand",
        )
        self.add_row_btn = pn.widgets.Button(
            name="Add row", button_type="primary", icon="plus"
        )
        self.clear_rows_btn = pn.widgets.Button(
            name="Clear rows", button_type="warning", icon="trash"
        )
        self.add_row_btn.on_click(self._handle_add_row)
        self.clear_rows_btn.on_click(self._handle_clear_rows)
        self.sync_axes_btn.on_click(self._handle_sync_axes)

        self.status_alert = pn.pane.Alert(
            "", alert_type="danger", visible=False, sizing_mode="stretch_width"
        )
        self.rows_container = pn.Column(sizing_mode="stretch_width")
        self.preview_panel = pn.Column(
            pn.pane.Markdown("### Current selection", sizing_mode="stretch_width"),
            pn.bind(
                self._render_current_selection,
                self.lr_slider,
                self.k_slider,
                self.batch_selector,
                self.comparison_selector,
                self.svd_window,
                self.loss_curve_toggle,
                self.epoch_batch_scale_toggle,
                self.raw_batch_scale_toggle,
                self.rank_scale_toggle,
            self.epoch_batch_x_range,
            self.epoch_batch_y_range,
            self.raw_batch_x_range,
            self.raw_batch_y_range,
            self.rank_x_range,
            self.rank_y_range,
            ),
        )

    def _init_layout(self) -> None:
        sidebar = [
            pn.pane.Markdown("### Pick a slice"),
            pn.pane.Markdown(
                "Select the SVD hyper-parameters that define a row of plots, "
                "optionally add baselines to compare, then click **Add row**. "
                "Each row displays the train/val curves across batch sizes "
                "plus the batch-wise losses and the number of non-zero singular values."
            ),
            self.lr_slider,
            self.k_slider,
            self.batch_selector,
            pn.pane.Markdown("### Compare against"),
            self.comparison_selector,
            pn.pane.Markdown("### Loss curve"),
            self.loss_curve_toggle,
            pn.pane.Markdown("### Epoch-scaled loss plot"),
            self.epoch_batch_scale_toggle,
            self.epoch_batch_x_range,
            self.epoch_batch_y_range,
            pn.pane.Markdown("### Raw batch loss plot"),
            self.raw_batch_scale_toggle,
            self.raw_batch_x_range,
            self.raw_batch_y_range,
            pn.pane.Markdown("### Rank plot"),
            self.rank_scale_toggle,
            self.rank_x_range,
            self.rank_y_range,
            self.svd_window,
            self.sync_axes_btn,
            pn.Row(self.add_row_btn, self.clear_rows_btn),
        ]

        main = [
            self.status_alert,
            self.preview_panel,
            pn.pane.Markdown(
                "Use the controls on the left to build as many comparison rows as you need."
            ),
            self.rows_container,
        ]

        self.template = pn.template.FastListTemplate(
            title="Toy 1D Scan Explorer",
            sidebar=sidebar,
            main=main,
        )

    # ---------------------------------------------------------------- callbacks
    def _handle_add_row(self, _event) -> None:
        config = self._current_row_config()
        try:
            row_panel = self._build_row_panel(config)
        except ValueError as exc:
            self.status_alert.object = str(exc)
            self.status_alert.alert_type = "danger"
            self.status_alert.visible = True
            return

        self.status_alert.visible = False
        self.rows_container.append(row_panel)
        self._select_row(row_panel)

    def _handle_clear_rows(self, _event) -> None:
        self.rows_container.clear()
        self.row_configs.clear()
        self.selected_row = None

    def _handle_sync_axes(self, _event) -> None:
        if not self.row_configs:
            return
        base = self._current_row_config()
        updates = dict(
            epoch_batch_scale=base.epoch_batch_scale,
            raw_batch_scale=base.raw_batch_scale,
            rank_scale=base.rank_scale,
            epoch_batch_x_range=base.epoch_batch_x_range,
            epoch_batch_y_range=base.epoch_batch_y_range,
            raw_batch_x_range=base.raw_batch_x_range,
            raw_batch_y_range=base.raw_batch_y_range,
            rank_x_range=base.rank_x_range,
            rank_y_range=base.rank_y_range,
        )
        for row_panel, cfg in list(self.row_configs.items()):
            new_cfg = replace(cfg, **updates)
            self._replace_row(row_panel, new_cfg)

    def _current_row_config(self) -> RowConfig:
        return self._config_from_values(
            self.lr_slider.value,
            self.k_slider.value,
            self.batch_selector.value,
            self.comparison_selector.value,
            self.svd_window.value,
            self.loss_curve_toggle.value,
            self.epoch_batch_scale_toggle.value,
            self.raw_batch_scale_toggle.value,
            self.rank_scale_toggle.value,
            self.epoch_batch_x_range.value,
            self.epoch_batch_y_range.value,
            self.raw_batch_x_range.value,
            self.raw_batch_y_range.value,
            self.rank_x_range.value,
            self.rank_y_range.value,
        )

    def _replace_row(self, row_panel: pn.Column, new_config: RowConfig) -> None:
        if row_panel not in self.rows_container.objects:
            return
        idx = self.rows_container.objects.index(row_panel)
        self.row_configs.pop(row_panel, None)
        new_panel = self._build_row_panel(new_config)
        self.rows_container.objects[idx] = new_panel
        if self.selected_row is row_panel:
            self.selected_row = new_panel
            self._update_row_highlights()

    def _set_controls_from_config(self, config: RowConfig) -> None:
        self.lr_slider.value = config.lr
        self.k_slider.value = config.k_fraction
        self.batch_selector.value = list(config.batch_sizes)
        self.comparison_selector.value = list(config.comparisons)
        self.svd_window.value = config.smoothing_window
        self.loss_curve_toggle.value = config.loss_curve
        self.epoch_batch_scale_toggle.value = config.epoch_batch_scale
        self.raw_batch_scale_toggle.value = config.raw_batch_scale
        self.rank_scale_toggle.value = config.rank_scale
        self.epoch_batch_x_range.value = self._format_range(config.epoch_batch_x_range)
        self.epoch_batch_y_range.value = self._format_range(config.epoch_batch_y_range)
        self.raw_batch_x_range.value = self._format_range(config.raw_batch_x_range)
        self.raw_batch_y_range.value = self._format_range(config.raw_batch_y_range)
        self.rank_x_range.value = self._format_range(config.rank_x_range)
        self.rank_y_range.value = self._format_range(config.rank_y_range)

    def _select_row(self, row_panel: pn.Column) -> None:
        config = self.row_configs.get(row_panel)
        if config is None:
            return
        self.selected_row = row_panel
        self._set_controls_from_config(config)
        self._update_row_highlights()

    def _update_row_highlights(self) -> None:
        for panel in list(self.row_configs.keys()):
            classes = list(panel.css_classes or ["svd-row"])
            classes = [cls for cls in classes if cls != "selected-row"]
            if panel is self.selected_row:
                classes.append("selected-row")
            panel.css_classes = classes

    def _config_from_values(
        self,
        lr,
        k_fraction,
        batch_sizes,
        comparisons,
        smoothing_window,
        loss_curve,
        epoch_batch_scale,
        raw_batch_scale,
        rank_scale,
        epoch_batch_x_range,
        epoch_batch_y_range,
        raw_batch_x_range,
        raw_batch_y_range,
        rank_x_range,
        rank_y_range,
    ) -> RowConfig:
        batch_tuple = tuple(sorted(batch_sizes)) or self.batch_sizes
        comparisons_tuple = tuple(comparisons or ())
        return RowConfig(
            k_fraction=float(k_fraction),
            lr=float(lr),
            batch_sizes=batch_tuple,
            comparisons=comparisons_tuple,
            smoothing_window=int(smoothing_window),
            loss_curve=str(loss_curve),
            epoch_batch_scale=str(epoch_batch_scale),
            raw_batch_scale=str(raw_batch_scale),
            rank_scale=str(rank_scale),
            epoch_batch_x_range=self._coerce_range(
                epoch_batch_x_range, (0.0, float(self.max_epochs))
            ),
            epoch_batch_y_range=self._coerce_range(
                epoch_batch_y_range, (float(self.loss_min), float(self.loss_max))
            ),
            raw_batch_x_range=self._coerce_range(
                raw_batch_x_range, (0.0, float(self.max_batches))
            ),
            raw_batch_y_range=self._coerce_range(
                raw_batch_y_range, (float(self.loss_min), float(self.loss_max))
            ),
            rank_x_range=self._coerce_range(
                rank_x_range, (0.0, float(self.max_batches))
            ),
            rank_y_range=self._coerce_range(
                rank_y_range, (0.0, float(self.rank_max) if self.rank_max else 1.0)
            ),
        )

    def _render_current_selection(
        self,
        lr,
        k_fraction,
        batch_sizes,
        comparisons,
        smoothing_window,
        loss_curve,
        epoch_batch_scale,
        raw_batch_scale,
        rank_scale,
        epoch_batch_x_range,
        epoch_batch_y_range,
        raw_batch_x_range,
        raw_batch_y_range,
        rank_x_range,
        rank_y_range,
    ):
        config = self._config_from_values(
            lr,
            k_fraction,
            batch_sizes,
            comparisons,
            smoothing_window,
            loss_curve,
            epoch_batch_scale,
            raw_batch_scale,
            rank_scale,
            epoch_batch_x_range,
            epoch_batch_y_range,
            raw_batch_x_range,
            raw_batch_y_range,
            rank_x_range,
            rank_y_range,
        )
        try:
            return self._build_row_panel(config, include_remove_button=False)
        except ValueError as exc:
            return pn.pane.Alert(
                str(exc), alert_type="danger", sizing_mode="stretch_width"
            )

    # --------------------------------------------------------------- build rows
    def _build_row_panel(
        self, config: RowConfig, *, include_remove_button: bool = True
    ) -> pn.Column:
        svd_entries, missing_batches = self._collect_svd_entries(config)
        if not svd_entries:
            raise ValueError(
                f"No SVD runs match batch sizes {config.batch_sizes} "
                f"for k_fraction={config.k_fraction:g} and lr={config.lr:g}."
            )

        epoch_scaled_fig, epoch_warn = self._build_epoch_scaled_batches(
            config, svd_entries=svd_entries
        )
        raw_batch_fig, raw_warn = self._build_raw_batch_losses(
            config, svd_entries=svd_entries
        )
        svd_fig = self._build_rank_plot(config, svd_entries=svd_entries)

        row_panel = pn.Column(
            sizing_mode="stretch_width", margin=(0, 0, 20, 0), css_classes=["svd-row"]
        )

        header_children = [
            pn.pane.Markdown(
                f"### k_frac = {config.k_fraction:g}, lr = {config.lr:g}  "
                f"| batches: {', '.join(str(bs) for bs in config.batch_sizes)}",
                sizing_mode="stretch_width",
            ),
            pn.layout.HSpacer(),
        ]
        if include_remove_button:
            remove_button = pn.widgets.Button(
                name="Remove row", button_type="light", icon="close"
            )
            apply_button = pn.widgets.Button(
                name="Apply current controls", button_type="primary", icon="refresh"
            )
            select_button = pn.widgets.Button(
                name="Select row", button_type="success", icon="hand-pointer"
            )

            def _remove_row(_event) -> None:
                if row_panel in self.rows_container.objects:
                    self.rows_container.objects.remove(row_panel)
                    self.row_configs.pop(row_panel, None)
                    if self.selected_row is row_panel:
                        self.selected_row = None
                        self._update_row_highlights()

            def _apply_row(_event) -> None:
                existing_config = self.row_configs.get(row_panel, config)
                change_flags = self._capture_axis_change_flags(existing_config)
                new_config = self._current_row_config()
                if existing_config:
                    new_config = self._merge_axis_settings(
                        existing_config, new_config, change_flags
                    )
                self._replace_row(row_panel, new_config)

            def _select(_event) -> None:
                self._select_row(row_panel)

            remove_button.on_click(_remove_row)
            apply_button.on_click(_apply_row)
            select_button.on_click(_select)
            header_children.extend([select_button, apply_button, remove_button])

        row_panel.append(pn.Row(*header_children, sizing_mode="stretch_width"))

        plot_row = pn.Row(
            pn.pane.Plotly(
                epoch_scaled_fig,
                height=420,
                sizing_mode="stretch_width",
                config={"displaylogo": False},
            ),
            pn.pane.Plotly(
                raw_batch_fig,
                height=420,
                sizing_mode="stretch_width",
                config={"displaylogo": False},
            ),
            pn.pane.Plotly(
                svd_fig,
                height=420,
                sizing_mode="stretch_width",
                config={"displaylogo": False},
            ),
            sizing_mode="stretch_width",
        )
        row_panel.append(plot_row)

        warnings = []
        if missing_batches:
            warnings.append(
                f"Missing SVD runs for batch sizes: {', '.join(str(bs) for bs in missing_batches)}."
            )
        warnings.extend(epoch_warn)
        warnings.extend(raw_warn)

        if warnings:
            row_panel.append(
                pn.pane.Alert(
                    "\n".join(warnings),
                    alert_type="warning",
                    sizing_mode="stretch_width",
                )
            )

        if include_remove_button:
            self.row_configs[row_panel] = config
            setattr(row_panel, "_config", config)
            if row_panel is self.selected_row:
                self._update_row_highlights()
        return row_panel

    # ----------------------------------------------------------- figure builders
    def _build_epoch_scaled_batches(
        self, config: RowConfig, *, svd_entries: Dict[int, pd.Series]
    ) -> Tuple[go.Figure, List[str]]:
        fig = self._base_figure(
            x_title="Epoch (from batches)",
            y_title=f"{config.loss_curve.title()} loss",
            y_scale=config.epoch_batch_scale,
            x_range=config.epoch_batch_x_range,
            y_range=config.epoch_batch_y_range,
        )
        warnings: List[str] = []
        epoch_key, batch_key = self._loss_keys(config.loss_curve)

        for bs, entry in svd_entries.items():
            losses = entry["losses"]
            color = self.batch_colors[bs]
            batches = losses.get(batch_key, [])
            if not batches:
                continue
            epoch_series = losses.get(epoch_key, [])
            num_epochs = max(1, len(epoch_series))
            epochs = np.linspace(0, num_epochs, num=len(batches))
            fig.add_scatter(
                x=epochs,
                y=batches,
                name=f"SVD {config.loss_curve} B{bs}",
                legendgroup=f"svd-{bs}",
                line=dict(color=color, width=2),
            )

        for compare in config.comparisons:
            compare_color = self.compare_colors[compare]
            for bs in config.batch_sizes:
                entry = self._fetch_standard_entry(
                    batch_size=bs, optimizer=compare.optimizer, lr=compare.lr
                )
                if entry is None:
                    warnings.append(
                        f"No {compare.optimizer} run for batch {bs} at lr={compare.lr:g}."
                    )
                    continue
                losses = entry["losses"]
                batches = losses.get(batch_key, [])
                if not batches:
                    continue
                epoch_series = losses.get(epoch_key, [])
                num_epochs = max(1, len(epoch_series))
                epochs = np.linspace(0, num_epochs, num=len(batches))
                label_base = f"{compare.optimizer} @ {compare.lr:g}  B{bs}"
                fig.add_scatter(
                    x=epochs,
                    y=batches,
                    name=label_base,
                    legendgroup=f"{label_base}-standard",
                    line=dict(color=compare_color, width=1.5),
                )

        if not fig.data:
            return (
                self._empty_figure("No epoch-scaled batch losses recorded."),
                warnings,
            )
        fig.update_layout(height=420)
        return fig, warnings

    def _build_raw_batch_losses(
        self, config: RowConfig, *, svd_entries: Dict[int, pd.Series]
    ) -> Tuple[go.Figure, List[str]]:
        fig = self._base_figure(
            x_title="Batch number",
            y_title=f"{config.loss_curve.title()} loss",
            y_scale=config.raw_batch_scale,
            x_range=config.raw_batch_x_range,
            y_range=config.raw_batch_y_range,
        )
        warnings: List[str] = []
        _, batch_key = self._loss_keys(config.loss_curve)
        for bs, entry in svd_entries.items():
            losses = entry["losses"]
            batches = losses.get(batch_key, [])
            if not batches:
                continue
            epochs = np.arange(1, len(batches) + 1)
            fig.add_scatter(
                x=epochs,
                y=batches,
                name=f"SVD {config.loss_curve} B{bs}",
                line=dict(color=self.batch_colors[bs]),
            )
        for compare in config.comparisons:
            compare_color = self.compare_colors[compare]
            for bs in config.batch_sizes:
                entry = self._fetch_standard_entry(
                    batch_size=bs, optimizer=compare.optimizer, lr=compare.lr
                )
                if entry is None:
                    warnings.append(
                        f"No {compare.optimizer} run for batch {bs} at lr={compare.lr:g}."
                    )
                    continue
                losses = entry["losses"]
                batches = losses.get(batch_key, [])
                if not batches:
                    continue
                epochs = np.arange(1, len(batches) + 1)
                label_base = f"{compare.optimizer} @ {compare.lr:g}  B{bs}"
                fig.add_scatter(
                    x=epochs,
                    y=batches,
                    name=label_base,
                    legendgroup=f"{label_base}-standard",
                    line=dict(color=compare_color, width=1.5),
                )
        if not fig.data:
            return self._empty_figure("No batch-wise losses recorded."), warnings
        fig.update_layout(height=420)
        return fig, warnings

    def _build_rank_plot(
        self, config: RowConfig, *, svd_entries: Dict[int, pd.Series]
    ) -> go.Figure:
        fig = self._base_figure(
            x_title="Batch",
            y_title="Avg. non-zero singular values",
            y_scale=config.rank_scale,
            x_range=config.rank_x_range,
            y_range=config.rank_y_range,
        )
        window = max(1, config.smoothing_window)
        for bs, entry in svd_entries.items():
            svd_info = entry.get("svd_info") or {}
            series = svd_info.get("num_nonzero_svs", [])
            if not series:
                continue
            smoothed = self._smooth_series(series, window=window)
            x_vals = np.arange(len(smoothed)) + 1
            fig.add_scatter(
                x=x_vals,
                y=smoothed,
                name=f"SVD rank  B{bs}",
                line=dict(color=self.batch_colors[bs]),
            )

        if not fig.data:
            return self._empty_figure("No SVD rank history recorded.")
        fig.update_layout(height=420)
        return fig

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _loss_keys(loss_curve: str) -> Tuple[str, str]:
        if str(loss_curve).lower() == "val":
            return "val", "val_batch"
        return "train", "train_batch"

    def _collect_svd_entries(
        self, config: RowConfig
    ) -> Tuple[Dict[int, pd.Series], List[int]]:
        entries: Dict[int, pd.Series] = {}
        missing: List[int] = []
        for bs in config.batch_sizes:
            entry = self._fetch_svd_entry(
                batch_size=bs, k_fraction=config.k_fraction, lr=config.lr
            )
            if entry is None:
                missing.append(bs)
            else:
                entries[bs] = entry
        return entries, missing

    def _fetch_svd_entry(
        self, *, batch_size: int, k_fraction: float, lr: float
    ) -> pd.Series | None:
        mask = (
            (self.svd_df["batch_size"] == batch_size)
            & np.isclose(self.svd_df["k_fraction"], k_fraction)
            & np.isclose(self.svd_df["lr"], lr)
        )
        if mask.any():
            return self.svd_df[mask].iloc[0]
        return None

    def _fetch_standard_entry(
        self, *, batch_size: int, optimizer: str, lr: float
    ) -> pd.Series | None:
        mask = (
            (self.standard_df["batch_size"] == batch_size)
            & np.isclose(self.standard_df["lr"], lr)
            & (self.standard_df["optimizer"] == optimizer)
        )
        if mask.any():
            return self.standard_df[mask].iloc[0]
        return None

    @staticmethod
    def _smooth_series(series: Sequence[float], window: int) -> List[float]:
        if window <= 1 or len(series) <= window:
            return list(series)
        kernel = np.ones(window) / window
        smoothed = np.convolve(series, kernel, mode="valid")
        return smoothed.tolist()

    @staticmethod
    def _base_figure(
        *,
        x_title: str,
        y_title: str,
        y_scale: str,
        x_range: Tuple[float, float] | None = None,
        y_range: Tuple[float, float] | None = None,
    ) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=70, r=40, t=60, b=80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
            ),
        )
        x_range_vals = list(x_range) if x_range else None
        fig.update_xaxes(
            title_text=x_title,
            showgrid=False,
            range=x_range_vals,
        )
        scale = "log" if str(y_scale).lower() == "log" else "linear"
        y_range_vals = None
        if y_range:
            lo, hi = y_range
            if scale == "log":
                lo = max(lo, 1e-12)
                hi = max(hi, lo * 1.001)
                y_range_vals = [math.log10(lo), math.log10(hi)]
            else:
                y_range_vals = [lo, hi]
        fig.update_yaxes(
            title_text=y_title,
            type=scale,
            showgrid=False,
            range=y_range_vals,
        )
        return fig

    @staticmethod
    def _parse_range(value: str | Tuple[float, float] | None) -> Tuple[float, float] | None:
        if value is None or value == "":
            return None
        if isinstance(value, tuple):
            return tuple(float(v) for v in value)
        if isinstance(value, str):
            parts = value.split(",")
            if len(parts) != 2:
                return None
            try:
                lo, hi = float(parts[0]), float(parts[1])
            except ValueError:
                return None
            return (lo, hi)
        try:
            lo, hi = value
            return (float(lo), float(hi))
        except Exception:
            return None

    def _coerce_range(
        self, value: str | Tuple[float, float] | None, default: Tuple[float, float]
    ) -> Tuple[float, float]:
        parsed = self._parse_range(value)
        if parsed is None:
            return tuple(float(v) for v in default)
        lo, hi = parsed
        if lo == hi:
            hi = lo + 1e-3
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    @staticmethod
    def _format_range(value: Tuple[float, float]) -> str:
        lo, hi = value
        return f"{lo:.6g},{hi:.6g}"

    def _capture_axis_change_flags(
        self, config: RowConfig | None
    ) -> Dict[str, bool]:
        if config is None:
            return {
                "epoch_scale": True,
                "epoch_x": True,
                "epoch_y": True,
                "raw_scale": True,
                "raw_x": True,
                "raw_y": True,
                "rank_scale": True,
                "rank_x": True,
                "rank_y": True,
            }
        return {
            "epoch_scale": self.epoch_batch_scale_toggle.value != config.epoch_batch_scale,
            "epoch_x": not self._ranges_equal(
                self.epoch_batch_x_range.value, config.epoch_batch_x_range
            ),
            "epoch_y": not self._ranges_equal(
                self.epoch_batch_y_range.value, config.epoch_batch_y_range
            ),
            "raw_scale": self.raw_batch_scale_toggle.value != config.raw_batch_scale,
            "raw_x": not self._ranges_equal(
                self.raw_batch_x_range.value, config.raw_batch_x_range
            ),
            "raw_y": not self._ranges_equal(
                self.raw_batch_y_range.value, config.raw_batch_y_range
            ),
            "rank_scale": self.rank_scale_toggle.value != config.rank_scale,
            "rank_x": not self._ranges_equal(
                self.rank_x_range.value, config.rank_x_range
            ),
            "rank_y": not self._ranges_equal(
                self.rank_y_range.value, config.rank_y_range
            ),
        }

    def _ranges_equal(
        self, widget_value: str | Tuple[float, float], config_range: Tuple[float, float]
    ) -> bool:
        parsed = self._parse_range(widget_value)
        if parsed is None:
            return False
        tol = 1e-9
        return all(abs(a - b) <= tol for a, b in zip(parsed, config_range))

    def _merge_axis_settings(
        self,
        old_config: RowConfig,
        new_config: RowConfig,
        change_flags: Dict[str, bool],
    ) -> RowConfig:
        updates = {}
        if not change_flags["epoch_scale"]:
            updates["epoch_batch_scale"] = old_config.epoch_batch_scale
        if not change_flags["epoch_x"]:
            updates["epoch_batch_x_range"] = old_config.epoch_batch_x_range
        if not change_flags["epoch_y"]:
            updates["epoch_batch_y_range"] = old_config.epoch_batch_y_range

        if not change_flags["raw_scale"]:
            updates["raw_batch_scale"] = old_config.raw_batch_scale
        if not change_flags["raw_x"]:
            updates["raw_batch_x_range"] = old_config.raw_batch_x_range
        if not change_flags["raw_y"]:
            updates["raw_batch_y_range"] = old_config.raw_batch_y_range

        if not change_flags["rank_scale"]:
            updates["rank_scale"] = old_config.rank_scale
        if not change_flags["rank_x"]:
            updates["rank_x_range"] = old_config.rank_x_range
        if not change_flags["rank_y"]:
            updates["rank_y_range"] = old_config.rank_y_range

        if updates:
            new_config = replace(new_config, **updates)
        return new_config

    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(size=14),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(template="plotly_white")
        return fig

    @staticmethod
    def _build_color_map(
        *, keys: Iterable, palette: Sequence[str]
    ) -> Dict:
        keys_list = list(keys)
        colors = {}
        for idx, key in enumerate(keys_list):
            colors[key] = palette[idx % len(palette)]
        return colors

    def _build_comparison_options(self) -> Dict[str, ComparisonConfig]:
        options: Dict[str, ComparisonConfig] = {}
        for optimizer in sorted(self.standard_df["optimizer"].unique()):
            subset = self.standard_df[self.standard_df["optimizer"] == optimizer]
            for lr in sorted(subset["lr"].unique()):
                label = f"{optimizer} @ {lr:g}"
                options[label] = ComparisonConfig(optimizer=optimizer, lr=float(lr))
        return options

    @staticmethod
    def _format_slider_options(values: Sequence[float]) -> Dict[str, float]:
        return {f"{float(v):g}": float(v) for v in values}

    def _compute_max_len(self, key: str) -> int:
        lengths = [
            len(losses.get(key, []))
            for losses in self.df["losses"]
            if key in losses and losses.get(key)
        ]
        return max(lengths or [1])

    def _compute_loss_bounds(self) -> Tuple[float, float]:
        mins, maxs = [], []
        for losses in self.df["losses"]:
            for key in ("train", "val", "train_batch", "val_batch"):
                series = losses.get(key, [])
                if series:
                    mins.append(min(series))
                    maxs.append(max(series))
        if not mins:
            return (1e-6, 1.0)
        lo = min(mins)
        hi = max(maxs)
        if lo == hi:
            hi = lo + 1e-3
        return lo, hi

    def _compute_rank_max(self) -> float:
        maxima = []
        for info in self.svd_df["svd_info"]:
            if not info:
                continue
            series = info.get("num_nonzero_svs", [])
            if series:
                maxima.append(max(series))
        if maxima:
            return max(maxima)
        return float(max(self.batch_sizes))


dashboard = Toy1DScanDashboard()
dashboard.panel.servable()
