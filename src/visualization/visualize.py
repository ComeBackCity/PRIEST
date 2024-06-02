import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram
import numpy as np

def show_clusters(clusters, data_files, dims=2, method='TSNE', data_path='./reports/figures/clustering/', show=False):
    for year_idx, cluster in enumerate(clusters):
        prot_vecs = cluster['data']
        labels = cluster['labels']

        if(method == 'TSNE'):
            pca_50 = PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(prot_vecs)
            reduced_data = TSNE(random_state=8, n_components=dims).fit_transform(pca_result_50)
        if(method == 'PCA'):
            pca = PCA(n_components=dims)
            reduced_data = pca.fit_transform(prot_vecs)
            print(f'Explained variance:{pca.explained_variance_ratio_}')  
            # reduced_centroids = NearestCentroid().fit(reduced_data, labels).centroids_

        fig = plt.figure()
        if (dims == 3): ax = fig.add_subplot(111, projection='3d')
        for i in range(len(reduced_data)):
            if (dims == 2):
                colors = 10 * ['r.', 'g.', 'y.', 'c.', 'm.', 'b.', 'k.']
                plt.plot(reduced_data[i][0], reduced_data[i][1], colors[labels[i]], markersize=10)
            if (dims == 3):
                colors = 10 * ['r', 'g', 'y', 'c', 'm', 'b', 'k']
                ax.scatter(reduced_data[i][0], reduced_data[i][1], reduced_data[i][2], c=colors[labels[i]], marker='.', zorder=1)
                # centroid = reduced_centroids[labels[i]]
                # ax.scatter(centroid[0], centroid[1], centroid[2], c='#0F0F0F', marker='x', zorder=100)

        plt.savefig(data_path + data_files[year_idx][:-4])

    if(show): plt.show()
    plt.close()

def show_dendogram(Z, year_idx, max_d=0.05, data_path='./reports/figures/dendograms/', show=False):
    fig = plt.figure()
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=0.05,
    )
    plt.savefig(data_path + str(year_idx))
    if(show): plt.show()
    plt.close()

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata