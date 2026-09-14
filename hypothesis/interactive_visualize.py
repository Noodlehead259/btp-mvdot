import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

data = np.load(
    project_root / "hypothesis" / "results" / "labels_sigma_025.npz"
)

clean_labels = data["clean_labels"]
noisy_labels = data["noisy_labels"]

k = 10

transition_matrix = np.zeros((k, k), dtype=np.int64)

for clean, noisy in zip(clean_labels, noisy_labels):
    transition_matrix[clean, noisy] += 1

row_totals = transition_matrix.sum(axis=1)

percentage_matrix = (
    transition_matrix / row_totals[:, None]
) * 100

stable_percentages = np.diag(percentage_matrix)

changed_percentages = 100 - stable_percentages

changed_count = np.sum(clean_labels != noisy_labels)
total_count = len(clean_labels)
disagreement = changed_count / total_count * 100

custom_data = np.empty(
    (k, k, 2),
    dtype=object
)

for i in range(k):
    for j in range(k):
        custom_data[i, j, 0] = transition_matrix[i, j]
        custom_data[i, j, 1] = percentage_matrix[i, j]

fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.64, 0.25],
    horizontal_spacing=0.18,
    subplot_titles=[
        "cluster transition matrix",
        "cluster instability"
    ]
)

heatmap = go.Heatmap(
    z=percentage_matrix,
    x=[str(i) for i in range(k)],
    y=[str(i) for i in range(k)],
    customdata=custom_data,
    text=np.round(percentage_matrix, 1),
    texttemplate="%{text}%",
    textfont={"size": 13},
    colorscale="Blues",
    zmin=0,
    zmax=100,
    colorbar={
        "title": "percentage",
        "x": 0.72,
        "xanchor": "center",
        "len": 0.82,
        "thickness": 18
    },
    hovertemplate=(
        "clean cluster: %{y}<br>"
        "noisy cluster: %{x}<br>"
        "samples: %{customdata[0]:,}<br>"
        "percentage: %{customdata[1]:.2f}%"
        "<extra></extra>"
    )
)

fig.add_trace(
    heatmap,
    row=1,
    col=1
)

bar = go.Bar(
    x=changed_percentages,
    y=[str(i) for i in range(k)],
    orientation="h",
    text=[f"{x:.2f}%" for x in changed_percentages],
    textposition="auto",
    hovertemplate=(
        "clean cluster: %{y}<br>"
        "changed: %{x:.2f}%"
        "<extra></extra>"
    )
)

fig.add_trace(
    bar,
    row=1,
    col=2
)

fig.update_xaxes(
    title_text="cluster after noise",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="cluster before noise",
    row=1,
    col=1
)

fig.update_xaxes(
    title_text="samples changed (%)",
    range=[0, max(changed_percentages) * 1.2],
    row=1,
    col=2
)

fig.update_yaxes(
    title_text="clean cluster",
    row=1,
    col=2
)

fig.update_layout(
    title={
        "text": (
            "cluster assignment instability under gaussian noise"
            f"<br><sup>"
            f"mnist training set | sigma = 0.25 | "
            f"{changed_count:,} / {total_count:,} samples changed "
            f"({disagreement:.2f}%)"
            f"</sup>"
        ),
        "x": 0.5
    },
    width=1250,
    height=800,
    showlegend=False
)

fig.write_html(
    project_root / "figures" / "figure_1_cluster_instability.html"
)

print("figure 1 saved.")
print("changed samples:", changed_count)
print("disagreement:", disagreement, "%")
print()
print("cluster instability:")

for i in range(k):
    print(
        f"cluster {i}: "
        f"{changed_percentages[i]:.2f}% changed"
    )