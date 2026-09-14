import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

data = np.load(
    project_root / "hypothesis" / "results" / "changed_samples_sigma_025.npz"
)

changed_indices = data["changed_indices"]
clean_images = data["clean_images"]
noisy_images = data["noisy_images"]
clean_labels = data["clean_labels"]
noisy_labels = data["noisy_labels"]

selected = []
seen_transitions = set()

for i in range(len(changed_indices)):
    transition = (
        int(clean_labels[i]),
        int(noisy_labels[i])
    )

    if transition not in seen_transitions:
        selected.append(i)
        seen_transitions.add(transition)

    if len(selected) == 12:
        break

if len(selected) < 12:
    for i in range(len(changed_indices)):
        if i not in selected:
            selected.append(i)

        if len(selected) == 12:
            break

fig = make_subplots(
    rows=4,
    cols=9,
    column_widths=[
        1, 1, 0.55,
        1, 1, 0.55,
        1, 1, 0.55
    ],
    horizontal_spacing=0.025,
    vertical_spacing=0.075,
    subplot_titles=[
        "clean", "noisy", "cluster change",
        "", "", "",
        "", "", ""
    ],
    specs=[
        [
            {"type": "heatmap"},
            {"type": "heatmap"},
            {"type": "xy"},
            {"type": "heatmap"},
            {"type": "heatmap"},
            {"type": "xy"},
            {"type": "heatmap"},
            {"type": "heatmap"},
            {"type": "xy"}
        ]
        for _ in range(4)
    ]
)

for position, i in enumerate(selected):
    row = position // 3 + 1
    group = position % 3

    base_col = group * 3 + 1

    clean_col = base_col
    noisy_col = base_col + 1
    transition_col = base_col + 2

    clean_image = clean_images[i]
    noisy_image = noisy_images[i]

    clean_cluster = int(clean_labels[i])
    noisy_cluster = int(noisy_labels[i])
    sample_index = int(changed_indices[i])

    fig.add_trace(
        go.Heatmap(
            z=clean_image,
            colorscale="gray",
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate=(
                f"sample index: {sample_index}<br>"
                f"clean cluster: {clean_cluster}"
                "<extra></extra>"
            )
        ),
        row=row,
        col=clean_col
    )

    fig.add_trace(
        go.Heatmap(
            z=noisy_image,
            colorscale="gray",
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate=(
                f"sample index: {sample_index}<br>"
                f"noisy cluster: {noisy_cluster}"
                "<extra></extra>"
            )
        ),
        row=row,
        col=noisy_col
    )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="text",
            text=[f"<b>{clean_cluster} → {noisy_cluster}</b>"],
            textfont={"size": 17},
            hovertext=[f"sample index: {sample_index}"],
            hoverinfo="text",
            showlegend=False
        ),
        row=row,
        col=transition_col
    )

    fig.update_xaxes(
        visible=False,
        range=[0, 27],
        row=row,
        col=clean_col
    )

    fig.update_yaxes(
        visible=False,
        range=[27, 0],
        scaleanchor=f"x{(row - 1) * 9 + clean_col}",
        scaleratio=1,
        row=row,
        col=clean_col
    )

    fig.update_xaxes(
        visible=False,
        range=[0, 27],
        row=row,
        col=noisy_col
    )

    fig.update_yaxes(
        visible=False,
        range=[27, 0],
        scaleanchor=f"x{(row - 1) * 9 + noisy_col}",
        scaleratio=1,
        row=row,
        col=noisy_col
    )

    fig.update_xaxes(
        visible=False,
        row=row,
        col=transition_col
    )

    fig.update_yaxes(
        visible=False,
        row=row,
        col=transition_col
    )

fig.update_layout(
    title={
        "text": (
            "representative samples with changed cluster assignments"
            "<br><sup>"
            "mnist training set | gaussian noise | sigma = 0.25"
            "</sup>"
        ),
        "x": 0.5,
        "y": 0.98
    },
    width=1400,
    height=850,
    margin={
        "l": 40,
        "r": 40,
        "t": 150,
        "b": 40
    },
    showlegend=False
)

for annotation in fig.layout.annotations:
    if annotation.text in ["clean", "noisy", "cluster change"]:
        annotation.font = {"size": 14}
        annotation.y = 1.02

fig.write_html(
    project_root / "figures" / "figure_2_changed_samples.html"
)

print("figure 2 saved.")

print()
print("selected samples:")

for i in selected:
    print(
        f"sample {changed_indices[i]}: "
        f"{clean_labels[i]} -> {noisy_labels[i]}"
    )