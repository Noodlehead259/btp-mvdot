import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

data = pd.read_csv(
    project_root /
    "hypothesis" /
    "results" /
    "noise_sweep.csv"
)

data["disagreement_percent"] = (
    data["disagreement_rate"] * 100
)

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=[
        "cluster disagreement",
        "normalized mutual information",
        "adjusted rand index"
    ]
)

fig.add_trace(
    go.Scatter(
        x=data["sigma"],
        y=data["disagreement_percent"],
        mode="lines+markers",
        marker={"size": 8},
        line={"width": 3},
        customdata=data[
            [
                "changed_images",
                "nmi",
                "ari"
            ]
        ],
        hovertemplate=(
            "sigma: %{x:.2f}<br>"
            "disagreement: %{y:.2f}%<br>"
            "changed samples: %{customdata[0]:,}<br>"
            "nmi: %{customdata[1]:.4f}<br>"
            "ari: %{customdata[2]:.4f}"
            "<extra></extra>"
        ),
        showlegend=False
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=data["sigma"],
        y=data["nmi"],
        mode="lines+markers",
        marker={"size": 8},
        line={"width": 3},
        hovertemplate=(
            "sigma: %{x:.2f}<br>"
            "nmi: %{y:.4f}"
            "<extra></extra>"
        ),
        showlegend=False
    ),
    row=2,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=data["sigma"],
        y=data["ari"],
        mode="lines+markers",
        marker={"size": 8},
        line={"width": 3},
        hovertemplate=(
            "sigma: %{x:.2f}<br>"
            "ari: %{y:.4f}"
            "<extra></extra>"
        ),
        showlegend=False
    ),
    row=3,
    col=1
)

fig.update_yaxes(
    title_text="disagreement (%)",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="nmi",
    row=2,
    col=1
)

fig.update_yaxes(
    title_text="ari",
    row=3,
    col=1
)

fig.update_xaxes(
    title_text="gaussian noise sigma",
    row=3,
    col=1
)

fig.update_layout(
    title={
        "text": (
            "cluster instability under increasing gaussian noise"
            "<br><sup>"
            "mnist training set | 60,000 samples"
            "</sup>"
        ),
        "x": 0.5
    },
    width=1100,
    height=900,
    margin={
        "l": 80,
        "r": 50,
        "t": 120,
        "b": 70
    },
    showlegend=False
)

fig.write_html(
    project_root /
    "figures" /
    "figure_3_noise_sensitivity.html"
)

print("figure 3 saved.")
print()
print(data.to_string(index=False))