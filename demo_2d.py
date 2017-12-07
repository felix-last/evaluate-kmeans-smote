# <codecell>
import os.path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
sns.set_style('whitegrid')
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from kmeans_smote import KMeansSMOTE
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
output_path = os.path.join(cfg['results_dir'], 'demo_2d')
if not os.path.exists(output_path):
    os.mkdir(output_path)

# <codecell>


def plot_before_after_oversampling(X, y, oversampler, dataset_name='', additional_text_after_oversampling=None, colors=['black', 'red', 'lightgreen']):
    """
    Plot dataset, perform k-means SMOTE and plot again.
    """
    oversampler_name, oversampler = oversampler
    X, y = np.asarray(X), np.asarray(y)

    # plot original data
    plt.scatter(
        x=X[:, 0],
        y=X[:, 1],
        c=np.asarray(colors)[y],
        linewidths=0)
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Before {}'.format(oversampler_name))
    plt.savefig('{}/{}-{}-before.png'.format(output_path, dataset_name, oversampler_name))
    plt.close()

    # oversample data
    X_ovs, y_ovs = oversampler.fit_sample(X, y)

    # plot oversampled data
    new_samples = np.where(np.isin(X_ovs, X, invert=True).all(axis=1))
    y_ovs[new_samples] = 2
    plt.scatter(
        x=X_ovs[:,0],
        y=X_ovs[:,1],
        c=np.asarray(colors)[y_ovs],
        linewidths=0)
    ax = plt.gca()
    ax.set_title('After {}'.format(oversampler_name))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if additional_text_after_oversampling is not None:
        ax.text(
            ax.get_xlim()[1] - (ax.get_xlim()[1] / 50),
            ax.get_ylim()[1] - (ax.get_ylim()[1] / 50),
            additional_text_after_oversampling,
            horizontalalignment='right',
            verticalalignment='top',
        )
    plt.savefig('{}/{}-{}-after.png'.format(output_path, dataset_name, oversampler_name))
    plt.close()

# <markdowncell>
# # Dataset A
# <codecell>
dataset_a = pd.read_csv('~/datasets/2d/a.csv', header=None)

# <markdowncell>
# ## Oversampling with SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    dataset_a.iloc[:, 0:2],
    dataset_a.iloc[:, 2],
    ('SMOTE', SMOTE()),
    'A'
)

# <markdowncell>
# ## Oversampling with k-means SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    dataset_a.iloc[:, 0:2],
    dataset_a.iloc[:, 2],
    ('k-means SMOTE', KMeansSMOTE(
        kmeans_args = {'n_clusters': 6},
        use_minibatch_kmeans = False
    )),
    'A',
    additional_text_after_oversampling='k = 6'
)


# <markdowncell>
# # Dataset B
# <codecell>
dataset_b = pd.read_csv('~/datasets/2d/b.csv', header=None)

# <markdowncell>
# ## Oversampling with SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    dataset_b.iloc[:, 0:2],
    dataset_b.iloc[:, 2],
    ('SMOTE', SMOTE()),
    'B'
)

# <markdowncell>
# ## Oversampling with k-means SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    dataset_b.iloc[:, 0:2],
    dataset_b.iloc[:, 2],
    ('k-means SMOTE', KMeansSMOTE(
        kmeans_args={'n_clusters': 3},
        use_minibatch_kmeans=False
    )),
    'B',
    additional_text_after_oversampling='k = 3'
)


# <markdowncell>
# # Dataset C
# <codecell>
dataset_c = pd.read_csv('~/datasets/2d/c.csv', header=None)

# <markdowncell>
# ## Oversampling with SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    dataset_c.iloc[:, 0:2],
    dataset_c.iloc[:, 2],
    ('SMOTE', SMOTE()),
    'C'
)

# <markdowncell>
# ## Oversampling with k-means SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    dataset_c.iloc[:, 0:2],
    dataset_c.iloc[:, 2],
    ('k-means SMOTE', KMeansSMOTE(
        kmeans_args={'n_clusters': 3},
        use_minibatch_kmeans=False
    )),
    'C',
    additional_text_after_oversampling='k = 3'
)


# <markdowncell>
# # Dataset Moons
# <codecell>
n_samples = 1500
moons_dataset = datasets.make_moons(n_samples=n_samples, noise=.3)
undersampler = RandomUnderSampler(ratio={0:200, 1:750})
moons_X, moons_y =undersampler.fit_sample(moons_dataset[0], moons_dataset[1])

# <markdowncell>
# ## Oversampling with SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    moons_X,
    moons_y,
    ('SMOTE', SMOTE()),
    'Moons'
)

# <markdowncell>
# ## Oversampling with k-means SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    moons_X,
    moons_y,
    ('k-means SMOTE', KMeansSMOTE(
        kmeans_args={'n_clusters': 50},
        use_minibatch_kmeans=False
    )),
    'Moons',
    additional_text_after_oversampling='k = 5'
)


# <markdowncell>
# # Dataset Circles
# <codecell>
n_samples = 1500
circles_dataset = datasets.make_circles(
    n_samples=n_samples, factor=.5, noise=.3)
undersampler = RandomUnderSampler(ratio={0: 300, 1: 750})
circles_X, circles_y = undersampler.fit_sample(
    circles_dataset[0], circles_dataset[1])

# <markdowncell>
# ## Oversampling with SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    circles_X,
    circles_y,
    ('SMOTE', SMOTE()),
    'Circles'
)

# <markdowncell>
# ## Oversampling with k-means SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    circles_X,
    circles_y,
    ('k-means SMOTE', KMeansSMOTE(
        kmeans_args={'n_clusters': 50},
        use_minibatch_kmeans=False
    )),
    'Circles',
    additional_text_after_oversampling='k = 5'
)
