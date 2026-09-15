from mat4py import loadmat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# 1. Species names for the 10 classes in the CUB subset
CLASS_NAMES = [
    "Black-footed Albatross",
    "Laysan Albatross",
    "Sooty Albatross",
    "Groove-billed Ani",
    "Crested Auklet",
    "Least Auklet",
    "Parakeet Auklet",
    "Rhinoceros Auklet",
    "Brewer Blackbird",
    "Red-winged Blackbird"
]

# 2. Load dataset and combine splits
data = loadmat('CUB.mat')
view1 = np.vstack([data['x1_train'], data['x1_test']])
view2 = np.vstack([data['x2_train'], data['x2_test']])
gt = np.concatenate([data['gt_train'], data['gt_test']]).reshape(-1)

# Convert 1-based indexing (1..10) to 0-based (0..9) if needed
if gt.min() == 1:
    gt = gt - 1

num_clusters = len(CLASS_NAMES)

# 3. Cluster each view independently
labels_v1 = KMeans(n_clusters=num_clusters, random_state=42, n_init=20).fit_predict(view1)
labels_v2 = KMeans(n_clusters=num_clusters, random_state=42, n_init=20).fit_predict(view2)

# 4. Map View 1 and View 2 clusters to ground-truth class names via Hungarian matching
def assign_class_names(pred_labels, true_labels, names):
    cm = confusion_matrix(pred_labels, true_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {row: names[col] for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[p] for p in pred_labels]), mapping

named_v1, map_v1 = assign_class_names(labels_v1, gt, CLASS_NAMES)
named_v2, map_v2 = assign_class_names(labels_v2, gt, CLASS_NAMES)

# 5. 2D embedding of View 1 manifold
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
v1_2d = tsne.fit_transform(view1)

# 6. Plotting
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Plot A: View 1 2D space labeled by View 1 matched class names
sns.scatterplot(
    x=v1_2d[:, 0], y=v1_2d[:, 1], hue=named_v1,
    hue_order=CLASS_NAMES, palette='tab10', ax=axes[0], s=40
)
axes[0].set_title("View 1 Manifold (Clusters Labeled by Best-Matched Class)")
axes[0].set_xlabel("t-SNE 1")
axes[0].set_ylabel("t-SNE 2")
axes[0].legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=True)

# Plot B: View 1 2D space labeled by View 2 matched class names
sns.scatterplot(
    x=v1_2d[:, 0], y=v1_2d[:, 1], hue=named_v2,
    hue_order=CLASS_NAMES, palette='tab10', ax=axes[1], s=40
)
axes[1].set_title("View 1 Manifold (View 2 Cluster Labels Projected)")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
axes[1].legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=True)

# Plot C: Confusion matrix indexed by class names
cm_named = confusion_matrix(named_v1, named_v2, labels=CLASS_NAMES)
sns.heatmap(
    cm_named, annot=True, fmt='d', cmap='Blues', ax=axes[2], cbar=False,
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
)
axes[2].set_title("Cross-View Cluster Agreement Heatmap")
axes[2].set_xlabel("View 2 Assigned Classes")
axes[2].set_ylabel("View 1 Assigned Classes")
plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.tight_layout()
plt.show()