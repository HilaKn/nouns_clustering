from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from abc import ABCMeta,abstractmethod, abstractproperty
import os
import operator
from commons import *

def get_cluster_from_factory(cluster_alg_name,words_data):
    if cluster_alg_name == "dbscan":
        return DBSCAN_Wrapper(words_data)
    elif cluster_alg_name == "kmeans":
        return KMeans_Wrapper(words_data)
    return

class ClusteringWrapper(object):

    __metaclass__ = ABCMeta

    def __init__(self,words_data):
        self.words_data = words_data
        self.labels = [-1]*words_data.dataset_size
        self._label_to_nouns = defaultdict(list)

    @abstractmethod
    def cluster_data(self,words_data):
        pass

    @abstractproperty
    def cluster_name(self):
        return "min_freq_{}".format(MIN_NOUN_FREQ)
        pass

    @property
    def label_to_nouns(self):
        if not self._label_to_nouns:
            for i,word in enumerate(self.words_data.words):
                label = self.labels[i]
                self._label_to_nouns[label].append(word)
        return self._label_to_nouns

    def output_clustering_results(self,output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_path = os.path.join(output_folder,self.cluster_name)
        sorted_labels_data = sorted(self.label_to_nouns.items(),key=operator.itemgetter(0))
        with open(file_path,'a') as f:
            for label,words_list in sorted_labels_data:
                if label== -1:
                    continue
                f.write("label [{}]\n".format(label))
                label_data_string = "\n".join(["{}\t{}".format(label,word) for word in words_list])
                f.write(label_data_string)
                f.write('\n')
            if self.label_to_nouns.has_key(-1):
                f.write("label [-1]")
                label_data_string = "\n".join(["{}\t{}".format(-1,word) for word in self.label_to_nouns[-1]])
                f.write(label_data_string)





class KMeans_Wrapper(ClusteringWrapper):

    K = 264

    def __init__(self,words_data):
        super(KMeans_Wrapper, self).__init__(words_data)


    @property
    def cluster_name(self):
        return super(KMeans_Wrapper, self).cluster_name + "_kmeans_k-{}".format(self.K)

    def cluster_data(self):
        print "KMEANS clustering for [{}] nouns".format(self.words_data.dataset_size)
        clustering_alg = KMeans(n_clusters =self.K, n_jobs=-1).fit(self.words_data.matrix)
        self.labels = clustering_alg.labels_
        print "Done KMEANS clustering"


class DBSCAN_Wrapper(ClusteringWrapper):

    EPS = 0.3
    MIN_SAMP = 5

    def __init__(self,words_data):
        super(DBSCAN_Wrapper, self).__init__(words_data)

    @property
    def cluster_name(self):
        return super(DBSCAN_Wrapper, self).cluster_name +"_dbscan_eps-{}_min_samp-{}.".format(self.EPS,self.MIN_SAMP)


    def cluster_data(self):
        print "DBSCAN clustering for [{}] nouns".format(self.words_data.dataset_size)
        clustering_alg = DBSCAN(eps =self.EPS,min_samples=self.MIN_SAMP, metric='cosine',
                                algorithm='brute',  n_jobs=1).fit(self.words_data.matrix)
        num_of_clusters = len(set(clustering_alg.labels_))-(1 if -1 in clustering_alg.labels_ else 0)
        print "DBSCAN found [{}] clusters".format(num_of_clusters)
        self.labels = clustering_alg.labels_

    # def output_clustering_results(self,output_folder):
    #
    #
    #     pass