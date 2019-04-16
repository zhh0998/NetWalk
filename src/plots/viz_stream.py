"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import operator
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
warnings.filterwarnings("ignore")

import matplotlib.pylab as pylab



params = {'font.family': 'sans-serif', 'font.serif': 'Times'}
pylab.rcParams.update(params)

def clustering_accuracy(membership,Labels):
    clusteringAcc=[]
    truemembership=membership
    predictedmembership={}
    for i in range(len(Labels)):
        community=Labels[i]
        if(community not in predictedmembership):
            predictedmembership[community]=[]
            predictedmembership[community].append(i)
        else:
            predictedmembership[community].append(i)
    for com in set(predictedmembership.keys()):
        count=0
        cluster={}
        clustermemebercount={}
        list_nodes = predictedmembership[com]
        for node in list_nodes:
            truecommunity=truemembership[node]
            if(truecommunity not in cluster):
                cluster[truecommunity] = []
                cluster[truecommunity].append(node)
            else:
                cluster[truecommunity].append(node)
            if(truecommunity not in clustermemebercount):
                clustermemebercount[truecommunity]=1
            else:
                clustermemebercount[truecommunity] = clustermemebercount[truecommunity]+1
            count=count+1
        maxkey=max(clustermemebercount.items(), key=operator.itemgetter(1))[0]
        maxClusterDensity=clustermemebercount[maxkey]
        clusteringaccuracy=maxClusterDensity/count
        clusteringAcc.append(clusteringaccuracy)
    overallAccuracy=min(clusteringAcc)
    return overallAccuracy


def viz_stream(G,rm_edges, fig, row, col, id, embedding_path="./tmp/embedding.txt",membership_path="./tmp/membership_karate.txt",representation_size=2,clusteringAccuracy=[]):
    """
    Visualizing karate graph for dynamical graph embedding demo
    :param rm_edges: removed edges
    :param fig: figure
    :param row: subplot row number
    :param col: subplot col number
    :param id: subplot id
    :param membership_path: node class in karate graph
    :param embedding_path: node embedding results
    :return: no return
    """
    #G1 = nx.karate_club_graph()

    membership = {}
    with open(membership_path, 'r') as member:
        for idx, line in enumerate(member):
            membership[idx] = int(line.strip())

    # https://networkx.github.io/documentation/development/_modules/networkx/drawing/layout.html
    pos = nx.fruchterman_reingold_layout(G, seed=42)
    #pos = nx.spring_layout(G)

    # http://matplotlib.org/mpl_examples/color/named_colors.pdf
    colors = ['orange', 'yellow', 'darkturquoise', 'green','red','blue','black']


    ff = fig.add_subplot(row, col, 2*(id-1)+1)
    ff.patch.set_visible(True)
    ff.axis('off')

    G.remove_edges_from(rm_edges)

    count = 0
    for com in set(membership.values()):
        list_nodes = [nodes for nodes in membership.keys() if membership[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=150, linewidths=0, node_color=colors[count])
        count += 1
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='k')#, font=font)
    nx.draw_networkx_edges(G, pos, alpha=0.5)


    fig.add_subplot(row, col, 2*(id-1)+2)
    ff.patch.set_visible(True)
    ff.axis('off')

    embedding = {}
    with open(embedding_path, 'r') as member:
        # member.readline()
        for line in member:
            res = line.strip().split()
            embedding[int(res[0])] = [float(res[1]), float(res[2])]
    count = 0
    for com in set(membership.values()):
        count += 1
        list_nodes = [nodes for nodes in membership.keys() if membership[nodes] == com]
        nx.draw_networkx_nodes(G, embedding, list_nodes, node_size=150, linewidths=0., node_color=colors[count])
    nx.draw_networkx_labels(G, embedding, font_size=9, font_color='k')  # , font=font)

    # region Clusetring Accuracy(Code in Progress)
    embedding = []
    with open(embedding_path, 'r') as member:
        # member.readline()
        for line in member:
            res = line.strip().split()
            embed_vector=[]
            for i in range(representation_size):
                embed_vector.append(float(res[i+1]))
            embedding.append(embed_vector)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(embedding)
    Labels = list(kmeans.labels_)

    clusteringAccuracy.append(clustering_accuracy(membership, Labels))

    # for i in range(len(Labels)):
    #     Labels[i]=Labels[i]+1
    # embedd_tsne = TSNE(n_components=2).fit_transform(embedding)
    # #plt.scatter(embedd_tsne[:len(embedd_tsne), 0], embedd_tsne[:len(embedd_tsne), 1], c=Labels[:len(embedd_tsne)], edgecolor='none',alpha=0.5, cmap=plt.get_cmap('jet', 10), s=5)
    # plt.scatter(embedd_tsne[:len(embedd_tsne), 0], embedd_tsne[:len(embedd_tsne), 1], c=Labels[:len(embedd_tsne)])
    # plt.colorbar()
    # plt.legend()
    # endregion
    return clusteringAccuracy


if __name__ == "__main__":
    # G = nx.path_graph(8)
    # nx.draw(G)
    # plt.show()

    membership_path = "../tmp/membership.txt"
    embedding_path = "../tmp/embedding.txt"
    viz_stream(membership_path, embedding_path)


