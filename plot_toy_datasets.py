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
output_path = os.path.join(cfg['results_dir'], 'toy_datasets')
if not os.path.exists(output_path):
    os.mkdir(output_path)

# <codecell>
def plot_before_after_oversampling(
    X, y, oversampler, dataset_name='',
    additional_text_after_oversampling=None,
    colors=['black', '#C1131D', '#527D37'],
    markers=['o', 'x', '+'], markersize=[20,30,50]):
    """
    Plot dataset, perform oversampling and plot again.
    """
    oversampler_name, oversampler = oversampler
    X, y = np.asarray(X), np.asarray(y)

    # plot original data
    for label in np.unique(y):
        plt.scatter(
            x=X[(y == label), 0],
            y=X[(y == label), 1],
            c=colors[label],
            marker=markers[label],
            linewidths= 0 if markers[label] == 'o' else None,
            s=markersize[label]
        )
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_title('Dataset {}'.format(dataset_name))
    plt.savefig(
        os.path.join(
            output_path,
            '{} before.png'.format(dataset_name).replace(' ', '_')
        ),
        bbox_inches="tight"
    )
    try:
        __IPYTHON__
        plt.show()
    except:
        plt.close()
    # oversample data
    X_ovs, y_ovs = oversampler.fit_sample(X, y)

    # plot oversampled data
    new_samples = np.where(np.isin(X_ovs, X, invert=True).all(axis=1))
    y_ovs[new_samples] = 2
    for label in np.unique(y_ovs):
        plt.scatter(
            x=X_ovs[(y_ovs == label), 0],
            y=X_ovs[(y_ovs == label), 1],
            c=colors[label],
            marker=markers[label],
            linewidths=0 if markers[label] == 'o' else None,
            s=markersize[label]
        )
    ax = plt.gca()
    ax.set_title(oversampler_name)
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
    plt.savefig(
        os.path.join(
            output_path,
            '{} {}.png'.format(dataset_name,oversampler_name).replace(' ', '_')
        ),
        bbox_inches="tight"
    )
    try:
        __IPYTHON__
        plt.show()
    except:
        plt.close()


# <markdowncell>
# # Dataset A
# <codecell>
dataset_a = pd.read_csv(os.path.join(cfg['dataset_dir'], 'a.csv'), header=None)

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
        kmeans_args={'n_clusters': 6},
        use_minibatch_kmeans=False
    )),
    'A',
    additional_text_after_oversampling='k = 6'
)


# <markdowncell>
# # Dataset B
# <codecell>
dataset_b = pd.read_csv(os.path.join(cfg['dataset_dir'], 'b.csv'), header=None)

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
dataset_c = pd.read_csv(os.path.join(cfg['dataset_dir'], 'c.csv'), header=None)

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
undersampler = RandomUnderSampler(ratio={0: 200, 1: 750})
moons_X, moons_y = undersampler.fit_sample(moons_dataset[0], moons_dataset[1])

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
