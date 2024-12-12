from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import math
import random
import numpy as np
from math import floor
from src.visualization import visualize
from src.utils import validation

from scipy.cluster.hierarchy import linkage, fcluster

def cluster_years(prot_vecs, method='DBSCAN'):
    clusters = []
    for year_prot_vecs in prot_vecs:

        if(method == 'DBSCAN'):
            min_samples = floor(len(year_prot_vecs)*0.01)
            clf = DBSCAN(eps=10, min_samples=min_samples, metric='euclidean').fit(year_prot_vecs)
            labels = clf.labels_
            centroids = NearestCentroid().fit(year_prot_vecs, labels).centroids_

        if(method == 'MeanShift'):
            clf = MeanShift().fit(year_prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_

        if(method == 'KMeans'):
            clf = KMeans(n_clusters=3)
            clf.fit(year_prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_

        clusters.append({'data':year_prot_vecs, 'labels':labels, 'centroids':centroids})

    return clusters

def squeeze_to_prot_vecs(trigram_vecs):
    prot_vecs = []
    print(len(trigram_vecs))
    for year_trigram_vecs in trigram_vecs:
        year_tv = np.array(year_trigram_vecs)
        print(year_tv.shape)
        year_trigram_vecs = year_tv.sum(axis=1)
        prot_vecs.append(year_trigram_vecs)

    print(prot_vecs[0].shape)
    return prot_vecs

def remove_outliers(data, clusters):
    for year_idx, cluster in enumerate(clusters):
        idxs_to_remove = []
        for i, label in enumerate(cluster['labels']):
            if(label == -1): idxs_to_remove.append(i) # -1 means outlier
        data[year_idx] = [prot_vec for i, prot_vec in enumerate(data[year_idx]) if i not in idxs_to_remove]
        clusters[year_idx]['labels'] = [label for i, label in enumerate(cluster['labels']) if i not in idxs_to_remove]
        clusters[year_idx]['data'] = [strain for i, strain in enumerate(cluster['data']) if i not in idxs_to_remove]
        if -1 in clusters[year_idx]['population']: del clusters[year_idx]['population'][-1]

    return data, clusters
        
def evaluate_clusters(clusters):
    scores = []
    for cluster in clusters:
        score = silhouette_score(cluster['data'], cluster['labels'])
        scores.append(score)

    average = sum(scores) / float(len(scores))
    return average

def link_clusters(clusters):
    no_years = len(clusters)
    neigh = NearestNeighbors(n_neighbors=2)

    for year_idx in range(no_years): 
        if(year_idx == no_years-1): # last year doesn't link
            clusters[year_idx]['links'] = [] 
            break 

        links = []
        current_centroids = clusters[year_idx]['centroids']
        next_year_centroids = clusters[year_idx+1]['centroids']
        neigh.fit(next_year_centroids)

        idxs_by_centroid = neigh.kneighbors(current_centroids, return_distance=False)

        for label in clusters[year_idx]['labels']:
            if (idxs_by_centroid[label][0] == -1): del idxs_by_centroid[label][0]
            links.append(idxs_by_centroid[label]) # centroid idx corresponds to label

        clusters[year_idx]['links'] = links

    return clusters

def label_encode(strains_by_year):
    amino_acids = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C', 'L', '-', 'B', 'J', 'Z', 'X']
    le = preprocessing.LabelEncoder()
    le.fit(amino_acids)


    encoded_strains = []
    for year_strains in strains_by_year:
        year_encoded_strains = []
        for strain in year_strains:
            chars = list(strain)
            year_encoded_strains.append(le.transform(chars))

        encoded_strains.append(year_encoded_strains)

    return encoded_strains

def cluster_raw(strains_by_year, method='dbscan'):
    clusters = []
    for year_idx, year_strains in enumerate(strains_by_year):
        min_samples = math.floor(len(year_strains)*0.05)
        if(method == 'dbscan'):
            clf = DBSCAN(eps=0.07, min_samples=min_samples, metric='hamming').fit(year_strains)
            # clf = DBSCAN(eps=10, min_samples=5, metric=metric).fit(year_strains)
            labels = clf.labels_

        if(method == 'hierarchy'):
            Z = linkage(year_strains, method='complete', metric='hamming')
            max_d = 0.135
            # visualize.show_dendogram(Z, year_idx, max_d)
            labels = fcluster(Z, 5, depth=10)
            # labels = fcluster(Z, max_d, criterion='distance')


        # print(year_strains)
        # print(labels)
        unique, count = np.unique(labels, return_counts=True)
        # print(unique, count)

        cluster = {
            'data': year_strains,
            'labels':labels, 
            'population': dict(zip(unique, count))} 

        clusters.append(cluster)
    return clusters

def sample_from_clusters(strains_by_year, clusters_by_years, sample_size, verbose=False):
    sampled_strains = [[]] * len(strains_by_year)

    # start sample from first cluster
    first_year_labels = clusters_by_years[0]['labels']
    first_year_population = clusters_by_years[0]['population']
    first_year_total = len(strains_by_year[0])

    for label_idx in first_year_population.keys(): # len(population) = no_clusters
        cluster_proportion = first_year_population[label_idx]/first_year_total
        cluster_sample_size = math.floor(sample_size*cluster_proportion)
        cluster_strains = [strains_by_year[0][i] for i, label in enumerate(first_year_labels) if label == label_idx]
        sampled_strains[0] = sampled_strains[0] + random.choices(cluster_strains, k=cluster_sample_size)

        # on last iteration sample missing
        missing_samples = sample_size - len(sampled_strains[0])
        if label_idx == list(first_year_population)[-1] and missing_samples > 0:
            print(f'Missing samples: {missing_samples}')
            sampled_strains[0] = sampled_strains[0] + random.choices(cluster_strains, k=missing_samples)


    # sample forward
    current_cluster = label_encode([sampled_strains[0]])[0]
    for year_idx in range(1, len(clusters_by_years)):
        year_clusters = clusters_by_years[year_idx]

        if(verbose): 
            print(f'\n>>> Linking {year_idx} year')
            print(f'Clusters\n{year_clusters["population"]}')

        neigh = NearestNeighbors(n_neighbors=1, metric='hamming')
        neigh.fit(year_clusters['data'])
        neighbour_strain_idx = neigh.kneighbors(current_cluster, return_distance=True)

        nice_neighs = [idx[0] for idx in neighbour_strain_idx[1]]
        links = [year_clusters['labels'][idx] for idx in nice_neighs]

        if(verbose): 
            validation.list_summary('Neighbours', nice_neighs)
            validation.list_summary('Links', links)

        clustered_strains = {}
        for label_idx in year_clusters['population'].keys():
            clustered_strains[label_idx] = [strains_by_year[year_idx][i] for i, label in enumerate(year_clusters['labels']) if label == label_idx]

        for link in links:
            sample = random.choice(clustered_strains[link])
            sampled_strains[year_idx] = sampled_strains[year_idx] + [sample]

    return sampled_strains
            

          
