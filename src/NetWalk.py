"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com> & Wenchao Yu
    Affiliation: NEC Labs America
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import matplotlib.pyplot as plt

from framework.imports import *
import framework.Model as MD

from datetime import datetime
import tensorflow as tf
import numpy as np
import networkx as nx

import warnings
from plots.viz_stream import viz_stream

from framework.netwalk_update import NetWalk_update

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def print_time():
    return datetime.now().strftime('[INFO %Y-%m-%d %H:%M:%S]')


def static_process(representation_size,walk_length,input,number_walks,init_percent,snap,output,datasetname):

    # region Parameters
    hidden_size = representation_size     # size of hidden codes to learn, default is 20
    activation = tf.nn.sigmoid
    rho = 0.5           # sparsity ratio
    lamb = 0.0017       # weight decay
    beta = 1            # sparsity weight
    gama = 340          # autoencoder weight
    walk_len = walk_length
    epoch = 400
    batch_size = 20     # number of epoch for optimizing, could be larger
    learning_rate = 0.1 # learning rate, for adam, using 0.01, for rmsprop using 0.1
    optimizer = "rmsprop"#"gd"#"rmsprop" #""lbfgs"#"rmsprop"#"adam"#"gd"#""lbfgs"#"adam"#
    corrupt_prob = [0]  # corrupt probability, for denoising AE
    # endregion

    # region STEP 1: Preparing data: training data and testing list of edges(for online updating)
    data_path = input
    netwalk = NetWalk_update(data_path, walk_per_node=number_walks,walk_len=walk_length, init_percent=init_percent, snap=snap)
    n = len(netwalk.vertices) # number of total nodes
    # endregion



    print("{} Number of nodes: {}".format(print_time(), n))
    print("{} Number of walks: {}".format(print_time(), number_walks))
    print("{} Data size (walks*length): {}".format(print_time(),number_walks*walk_length))
    print("{} Generating network walks...".format(print_time()))
    print("{} Clique embedding training...".format(print_time()))


    dimension = [n, hidden_size]

    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,epoch, batch_size, learning_rate, optimizer, corrupt_prob)

    init_edges, snapshots,edges = netwalk.data
    edges = edges-1   #karate
    # G = nx.Graph()
    # G.add_edges_from(edges)
    # vertices = np.unique(edges)
    G = nx.Graph()
    G.add_edges_from(edges)
    clusteringAccuracy=[]

    edge_list = tuple(map(tuple, edges))

    data = netwalk.getInitWalk()

    fig = plt.figure(figsize=(12, 12))

    # STEP 2: Learning initial embeddings for training edges
    embeddings=embedding_code(embModel, data, n, output)


    # list of initial edge list tuples
    tuples = tuple(map(tuple, init_edges-1))#karate
    #tuples = tuple(map(tuple, init_edges))

    # complementary set of edges for initial edges
    rm_list = [x for x in edge_list if x not in tuples]

    # visualize initial embedding

    clusteringAccuracy=viz_stream(G,rm_list, fig, 5, 2, 1,output,"./tmp/membership_"+datasetname+".txt",representation_size,clusteringAccuracy)

    # STEP 3: over different snapshots of edges, dynamically updating embeddings of nodes and conduct
    #         online anomaly detection for edges, visualize the anomaly score of each snapshot
    snapshotNum = 0
    while(netwalk.hasNext()):
        G = nx.Graph()
        G.add_edges_from(edges)
        data = netwalk.nextOnehotWalks()
        tuples = tuple(map(tuple, snapshots[snapshotNum] - 1)) + tuples
        snapshotNum += 1
        embedding_code(embModel, data, n,output)
        rm_list = [x for x in edge_list if x not in tuples]
        clusteringAccuracy=viz_stream(G,rm_list, fig, 5, 2, snapshotNum+1,output,"./tmp/membership_"+datasetname+".txt",representation_size)
        print(clusteringAccuracy)


    #plt.show()

    fig.savefig('../plots/graph_'+datasetname+'.png')
    f = open('./tmp/accuracy_' + datasetname + '.txt', 'a+')
    f.write("dimension is "+str(dimension))
    f.write("\n")
    for acc in clusteringAccuracy:
        f.write(str(acc))
        f.write("\n")
    f.write("\n")
    f.write("\n")
    #np.savetxt(f, clusteringAccuracy, fmt="%g")

    print("finished")





def embedding_code(model, data, n,output):
    """
            function embedding_code(model, data, n, args)
            #  the function feed ''data'' which is a list of walks
            #  the embedding ''model'', n: the number of total nodes
            return: the embeddings of all nodes, each row is one node, each column is one dimension of embedding
        """
    # STEP 2:  optimizing to fit parameter learning
    model.fit(data)

    # STEP 3: retrieve the embeddings
    node_onehot = np.eye(n)
    res = model.feedforward_autoencoder(node_onehot)
    ids = np.transpose(np.array(range(n)))
    ids = np.expand_dims(ids, axis=1)
    embeddings = np.concatenate((ids, res), axis=1)

    # STEP 4: save results
    np.savetxt(output, embeddings, fmt="%g")
    print("{} Done! Embeddings are saved in \"{}\"".format(print_time(),output))
    return embeddings


def main():
    format='adjlist'
    snap=2
    init_percent=0.5
    #datasetname = 'karate'
    #input='../data/karate.edges'
    # datasetname = 'dolphin'
    # input = '../data/dolphins.mtx'
    #datasetname = 'cora'
    #input = '../data/cora.edgelist'
    # datasetname = 'citeseer'
    # input = '../data/citeseer.edgelist'
    datasetname = 'toy'
    input = '../data/toy.edges'
    number_walks=10
    output = './tmp/embedding_' + datasetname + '.txt'
    representation_size=8
    seed=24
    walk_length=3
    static_process(representation_size,walk_length,input,number_walks,init_percent,snap,output,datasetname)
if __name__ == "__main__":
    main()
