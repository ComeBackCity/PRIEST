import sys, os
sys.path.append('./')
from src.utils import utils
from src.data import cluster
from src.visualization import visualize
from src.data import make_dataset
import pandas as pd

# data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
# data_files = ['2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_files = ['2016.csv']
data_path = './data/raw/'

# clustering_methods = ['DBSCAN', 'KMeans', 'MeanShift']
reduction_method = 'TSNE'
reduction_method = 'PCA'
visualized_dimensions = 3 
clustering_methods = ['DBSCAN']

trigram_vecs, _, _, _ = utils.read_and_process_to_trigram_vecs(data_files, data_path, sample_size=0, squeeze=False)
print(trigram_vecs)
prot_vecs = cluster.squeeze_to_prot_vecs(trigram_vecs)

# concated_years = [[]]
# for year_trigram_vecs in trigram_vecs:
#     concated_years[0] += year_trigram_vecs
# trigram_vecs = concated_years


print(f'Shape: {len(prot_vecs)}x{len(prot_vecs[0])}x{len(prot_vecs[0][0])}')

for method in clustering_methods:
    clusters_by_year = cluster.cluster_years(prot_vecs, method)
    print(f'Number of clusters in the first year: {len(clusters_by_year[0]["centroids"])}')
    # average = cluster.evaluate_clusters(clusters_by_year)
    # print(f'Average variance of {method}: {average}')

    clusters_by_year = cluster.link_clusters(clusters_by_year)

    # save
    # strains_by_year = make_dataset.read_strains_from(data_files, data_path)
    # for i, clusters in enumerate(clusters_by_year):
    #     path = data_files[i]

    #     df = pd.read_csv(data_path + path)
    #     df['cluster'] = clusters['labels']

    #     df['links'] = pd.Series(clusters['links'])

    #     path = f'./data/interim/{method}.{path}'
    #     print(f'saving to : {path}')
    #     df.to_csv(path)
    clusters_by_year = cluster.remove_outliers(clusters_by_year)
    visualize.show_clusters(clusters_by_year, method=reduction_method, dims=visualized_dimensions)
