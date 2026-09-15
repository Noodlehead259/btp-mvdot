from mat4py import loadmat
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

data = loadmat('CUB.mat')

view1 = np.vstack([data['x1_train'], data['x1_test']])
view2 = np.vstack([data['x2_train'], data['x2_test']])
k = len(np.unique(np.concatenate([data['gt_train'], data['gt_test']])))

labels_v1 = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(view1)
labels_v2 = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(view2)

print(adjusted_rand_score(labels_v1, labels_v2))