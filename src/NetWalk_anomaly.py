"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com> & Wenchao Yu
    Affiliation: NEC Labs America
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

from framework.imports import *
import framework.Model as MD
from framework.anomaly_generation import anomaly_generation
from framework.load_email_eu import load_email_eu
from framework.anomaly_detection import anomaly_detection
from framework.anomaly_detection_stream import anomaly_detection_stream
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import tensorflow as tf
import numpy as np
import plots.DynamicUpdate as DP

import warnings
from framework.netwalk_update import NetWalk_update

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def print_time():
    return datetime.now().strftime('[INFO %Y-%m-%d %H:%M:%S]')


def static_process(representation_size,walk_length,input,number_walks,init_percent,snap,output,datasetname):
    # region Preprocess the data(change directed to undirected/remove self loops/remove duplicate edges)
    sample_rate = 1     #0.5
    data, n, m = load_email_eu(input, sample_rate)
    # endregion

    # region Parameters
    hidden_size = representation_size                 # size of hidden codes to learn, default is 20
    dimension = [n, hidden_size]
    activation = tf.nn.sigmoid
    rho = 0.5                                         # sparsity ratio
    lamb = 0.0017                                     # weight decay
    beta = 1                                          # sparsity weight
    gama = 340                                        # autoencoder weight
    walk_len = walk_length                            # Length of each walk
    epoch = 30                                        # number of epoch for optimizing, could be larger
    batch_size = 40                                   # should be smaller or equal to args.number_walks*n
    learning_rate = 0.01                              # learning rate, for adam, using 0.01, for rmsprop using 0.1
    optimizer = "adam"                                #"rmsprop"#"gd"#"rmsprop" #"""gd"#""lbfgs"
    corrupt_prob = [0]                                # corrupt probability, for denoising AE
    ini_graph_percent = init_percent                  # percent of edges in the initial graph
    alfa = 0.01 #0.5(paper)                           # updating parameter for online k-means to update clustering centroids
    if(datasetname=="karate"):
        anomaly_percent = 0.1
        k=4
    elif(datasetname=="toy"):
        anomaly_percent = 1
        k=2
    elif(datasetname=="cora"):
        anomaly_percent = 0.1
        k=7
    elif(datasetname=="citeseer"):
        anomaly_percent = 0.1
        k = 6
    elif (datasetname == "dolphin"):
        anomaly_percent = 0.1
        k = 3

    print("No of Clusters in Dataset "+str(datasetname)+" is "+str(k))
    # endregion

    # region STEP 1: Generates Anomaly data: training data and testing list of edges(for online updating)
    membership_path="./tmp/membership_"+datasetname+".txt"
    #synthetic_test, train_mat, train = anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m,membership_path)
    synthetic_test, train_mat, train = anomaly_generation(0.7, anomaly_percent, data, n, m,membership_path)
    data_zip = []
    data_zip.append(synthetic_test)
    data_zip.append(train)
    # endregion

    # region generating initial training walks
    netwalk = NetWalk_update(data_zip, walk_per_node=number_walks, walk_len=walk_length,init_percent=init_percent, snap=snap)
    ini_data = netwalk.getInitWalk()
    print(np.shape(ini_data[0]))
    # endregion

    # region Initialise Model
    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,epoch, batch_size, learning_rate, optimizer, corrupt_prob)
    # endregion

    # region STEP 2: Learning initial embeddings for training edges
    embedding = getEmbedding(embModel, ini_data, n)
    # endregion


    # region conduct anomaly detection using first snapshot of testing edges
    areaUnderCurve=[]
    xValue=[]
    test_piece=synthetic_test[0:snap, :]
    scores, auc, n0, c0, res, ab_score = anomaly_detection(embedding, train, test_piece, k)
    areaUnderCurve.append(auc)
    xValue.append(0)
    #scores, auc, n0, c0, res, ab_score = anomaly_detection(embedding, train, synthetic_test, k)
    print('initial auc of anomaly detection:', auc)
    print('initial anomaly score:', ab_score)
    # endregion

    # region Online Increment
    # STEP 3: over different snapshots of edges, dynamically updating embeddings of nodes and conduct
    #         online anomaly detection for edges, visualize the anomaly score of each snapshot
    snapshotNum = 1
    while (netwalk.hasNext()):
        # region Include next walks dynamically and find embedding
        snapshot_data = netwalk.nextOnehotWalks()
        embedding = getEmbedding(embModel, snapshot_data, n)
        # endregion
        if netwalk.hasNext():
            if len(synthetic_test) > snap * (snapshotNum + 1):
                #test_piece = synthetic_test[snap * snapshotNum:snap * (snapshotNum + 1), :]
                test_piece = synthetic_test[:snap * (snapshotNum + 1), :]
            else:
                test_piece = synthetic_test
        # online anomaly detection, each execution will update the clustering center
        scores, auc, n0, c0, res, ab_score = anomaly_detection_stream(embedding, train, test_piece, k, alfa, n0, c0)
        print('auc of anomaly detection at snapshot %d: %f' % (snapshotNum, auc))
        print('anomaly score at snapshot %d: %f' % (snapshotNum, ab_score))
        areaUnderCurve.append(auc)
        xValue.append(snapshotNum)
        snapshotNum += 1
    plt.plot(xValue, areaUnderCurve)
    plt.savefig('../plots/anomalyaccuracy_' + datasetname +str(datetime.datetime.now())+'.png')
    # endregion


def getEmbedding(model, data, n):
    """
        function getEmbedding(model, data, n)
        #  the function feed ''data'' which is a list of walks
        #  the embedding ''model'', n: the number of total nodes
        return: the embeddings of all nodes, each row is one node, each column is one dimension of embedding
    """
    # batch optimizing to fit the model
    model.fit(data)

    # Retrieve the embeddings
    node_onehot = np.eye(n)
    res = model.feedforward_autoencoder(node_onehot)
    return res


def main():
    # region Parameter Initialise
    init_percent = 0.5
    datasetname=sys.argv[1]
    #datasetname = 'karate'
    # datasetname = 'dolphin'
    #
    # datasetname = 'cora'
    #
    # datasetname = 'citeseer'
    #
    # datasetname = 'toy'
    number_walks = 20
    output = './tmp/embedding_' + datasetname + '.txt'
    if (datasetname == "karate"):
        input = '../data/karate.edges'
        snap = 10
        representation_size = 32
    elif (datasetname == "toy"):
        input = '../data/toy.edges'
        snap = 2
        representation_size = 8
    elif (datasetname == "dolphin"):
        input = '../data/dolphins.mtx'
        snap = 10
        representation_size = 32
    elif (datasetname == "cora"):
        input = '../data/cora.edgelist'
        snap = 400
        representation_size = 64
    elif (datasetname == "citeseer"):
        input = '../data/citeseer.edgelist'
        snap = 600
        representation_size = 128
    walk_length = 3
    # endregion

    static_process(representation_size, walk_length, input, number_walks, init_percent, snap, output, datasetname)


if __name__ == "__main__":
    main()
