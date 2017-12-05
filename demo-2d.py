# <codecell>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from imblearn.over_sampling import SMOTE
from kmeans_smote import KMeansSMOTE
%matplotlib inline

def plot_before_after_oversampling(dataset, oversampler, colors=['black','red','lightgreen']):
    """
    Load a dataset, plot it, perform k-means SMOTE and plot again.
    """
    oversampler_name, oversampler = oversampler

    # plot original data
    df_a = pd.read_csv(dataset, header=None)
    df_a.plot.scatter(
        x=0,
        y=1,
        c=np.asarray(colors)[df_a[2]],
        linewidths=0)
    plt.gca().set_title('Before {}'.format(oversampler_name))
    plt.show()

    # oversample data
    X, y = oversampler.fit_sample(df_a.loc[:, 0:1], df_a.loc[:, 2])

    # plot oversampled data
    new_samples = np.where(np.isin(X, df_a.loc[:, 0:1], invert=True).all(axis=1))
    y[new_samples] = 2
    plt.scatter(
        x=X[:,0],
        y=X[:,1],
        c=np.asarray(colors)[y],
        linewidths=0)
    plt.gca().set_title('After {}'.format(oversampler_name))
    plt.show()

# <markdowncell>
# # Dataset A

# <markdowncell>
# ## Oversampling with k-means SMOTE

# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    '~/datasets/2d/a.csv',
    ('k-means SMOTE', KMeansSMOTE(
        kmeans_args = {'n_clusters': 3},
        use_minibatch_kmeans = False
    ))
)

# <markdowncell>
# ## Oversampling with SMOTE

# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    '~/datasets/2d/a.csv',
    ('SMOTE', SMOTE())
)

# <markdowncell>
# # Dataset B

# <markdowncell>
# ## Oversampling with k-means SMOTE

# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    '~/datasets/2d/b.csv',
    ('k-means SMOTE', KMeansSMOTE(
        kmeans_args={'n_clusters': 3},
        use_minibatch_kmeans=False
    ))
)

# <markdowncell>
# ## Oversampling with SMOTE

# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    '~/datasets/2d/b.csv',
    ('SMOTE', SMOTE())
)


# <markdowncell>
# # Dataset C

# <markdowncell>
# ## Oversampling with k-means SMOTE
# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    '~/datasets/2d/c.csv',
    ('k-means SMOTE', KMeansSMOTE(
        kmeans_args={'n_clusters': 3},
        use_minibatch_kmeans=False
    ))
)

# <markdowncell>
# ## Oversampling with SMOTE

# <codecell>
np.random.seed(1)
plot_before_after_oversampling(
    '~/datasets/2d/c.csv',
    ('SMOTE', SMOTE())
)
