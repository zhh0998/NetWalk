import logging
from framework.imports import *
import framework.Model as MD
from framework.anomaly_generation import anomaly_generation
from framework.load_email_eu import load_email_eu
from framework.anomaly_detection import anomaly_detection
from framework.anomaly_detection_stream import anomaly_detection_stream
import matplotlib.pyplot as plt
import queue
import random
import datetime
import tensorflow as tf
import numpy as np
import operator
import warnings
from framework.netwalk_update import NetWalk_update
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def print_time():
    return datetime.now().strftime('[INFO %Y-%m-%d %H:%M:%S]')

def processEdges(fake_edges, data):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)

    tmp = fake_edges[idx_fake]
    tmp[:, [0, 1]] = tmp[:, [1, 0]]

    fake_edges[idx_fake] = tmp

    idx_remove_dups = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] < 0)

    fake_edges = fake_edges[idx_remove_dups]
    a = fake_edges.tolist()
    b = data.tolist()
    c = []

    for i in a:
        if i not in b:
            c.append(i)
    #fake_edges = np.array(c)
    fake_edges = c
    uniqueEdge=[]
    for edge in fake_edges:
        if edge in uniqueEdge:
            continue
        else:
            uniqueEdge.append(edge)
    return np.array(uniqueEdge)

def preprocessGraph(data,init_percent,n,m):
    partGraphSize = int(len(data) * init_percent)
    nodes = np.unique(data)
    adjacentVertices = queue.Queue(nodes.size)
    # region Record Degree in dict and sort dict based on value
    degree = {}
    for edge in data:
        src = edge[0]
        dest = edge[1]
        if src in degree:
            degree[src] = degree[src] + 1
        else:
            degree[src] = 1
        if dest in degree:
            degree[dest] = degree[dest] + 1
        else:
            degree[dest] = 1
    sdegree = sorted(degree.items(), key=operator.itemgetter(1))
    # endregion

    # region Select edges of the spanning tree
    source = sdegree[0][0]
    adjacentVertices.put(source)
    spanningEdges = []
    bfs = []
    seenNode = []
    seenNode.append(source)
    while (adjacentVertices.qsize() != 0):
        current = adjacentVertices.get()
        bfs.append(current)
        edgeList = []
        for edge in data:
            src = edge[0]
            dest = edge[1]
            if (src == current or dest == current):
                edgeList.append(edge)
        adj = list(np.unique(np.array(edgeList)))
        for nodee in adj:
            if nodee not in bfs and nodee not in seenNode:
                adjacentVertices.put(nodee)
                seenNode.append(nodee)
                for edge in edgeList:
                    if (nodee == edge[0] or nodee == edge[1]):
                        spanningEdges.append(list(edge))
    partGraph = spanningEdges
    # endregion

    # region More Edges to fill rest of the edges for The Graph to use
    while (partGraphSize > len(spanningEdges)):
        randindx = random.randint(1, np.shape(data)[0] - 1)
        edge = data[randindx]
        if (list(edge) not in partGraph):
            partGraph.append(list(edge))
    # endregion

    # Remaining Edges
    # region Remaining Positive Edges
    remainingEdges = []
    for edge in data:
        if list(edge) not in partGraph:
            remainingEdges.append(list(edge))
    positiveedges = remainingEdges
    # endregion

    # region Generate Random Edges By Combination of nodes
    idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) + 1, axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) + 1, axis=1)
    generate_edges = np.concatenate((idx_1, idx_2), axis=1)
    # remove self-loops and duplicates and order fake edges
    fake_edges = processEdges(generate_edges, data)
    # endregion

    # region Negative Edges
    negativeEdges = []
    for i in range(len(remainingEdges)):
        randindx = random.randint(1, np.shape(fake_edges)[0] - 1)
        negativeEdges.append(list(fake_edges[randindx]))
    # endregion

    totalEdge = []
    for edges in negativeEdges:
        totalEdge.append(edges)
    for edges in positiveedges:
        totalEdge.append(edges)
    random.shuffle(totalEdge)
    truelabel = []
    for edge in totalEdge:
        if edge in positiveedges:
            truelabel.append(1)
        elif edge in negativeEdges:
            truelabel.append(0)
    (trainData, testData, trainLabels, testLabels) = train_test_split(totalEdge, truelabel, test_size=0.2,
                                                                      random_state=42)
    return partGraph, trainData, trainLabels, testData, testLabels

def linkPrediction(embedding,trainData,trainLabels,testData,testLabels):
    src = embedding[trainData[:, 0] - 1, :]
    dst = embedding[trainData[:, 1] - 1, :]
    test_src = embedding[testData[:, 0] - 1, :]
    test_dst = embedding[testData[:, 1] - 1, :]
    encoding_method = 'Hadamard'
    if encoding_method == 'Average':
        codes = (src + dst) / 2
        test_codes = (test_src + test_dst) / 2
    elif encoding_method == 'Hadamard':
        codes = np.multiply(src, dst)
        test_codes = np.multiply(test_src, test_dst)
    elif encoding_method == 'WeightedL1':
        codes = abs(src - dst)
        test_codes = abs(test_src - test_dst)
    elif encoding_method == 'WeightedL2':
        codes = (src - dst) ** 2
        test_codes = (test_src - test_dst) ** 2

    # knn = KNeighborsClassifier()
    # knn.fit(codes, trainLabels)
    # ypredict=knn.predict(codes)
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class = 'multinomial').fit(codes,trainLabels)
    ypredict=clf.predict(test_codes)
    accuracy=0
    for i in range(len(testLabels)):
        if testLabels[i]==ypredict[i]:
            accuracy=accuracy+1
    return accuracy/len(testLabels)


def static_process(representation_size,walk_length,input,number_walks,init_percent,snap,output,datasetname):
    # region Preprocess the data(change directed to undirected/remove self loops/remove duplicate edges)
    sample_rate = 1     #0.5
    data, n, m = load_email_eu(input, sample_rate)
    GraphEdges, trainData, trainLabels, testData, testLabels=preprocessGraph(data,0.7,n,m)
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
    epoch = 100                                        # number of epoch for optimizing, could be larger
    batch_size = 40                                   # should be smaller or equal to args.number_walks*n
    learning_rate = 0.01                              # learning rate, for adam, using 0.01, for rmsprop using 0.1
    optimizer = "adam"                                #"rmsprop"#"gd"#"rmsprop" #"""gd"#""lbfgs"
    corrupt_prob = [0]                                # corrupt probability, for denoising AE
    ini_graph_percent = init_percent                  # percent of edges in the initial graph
    alfa = 0.01 #0.5(paper)                           # updating parameter for online k-means to update clustering centroids
    if(datasetname=="karate"):
        anomaly_percent = 0.3
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


    # region generating initial training walks
    netwalk = NetWalk_update(data, walk_per_node=number_walks, walk_len=walk_length,init_percent=init_percent, snap=snap)
    ini_data = netwalk.getInitWalk()
    # endregion

    # region Initialise Model
    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,epoch, batch_size, learning_rate, optimizer, corrupt_prob)
    # endregion

    # region STEP 2: Learning initial embeddings for training edges
    embedding = getEmbedding(embModel, ini_data, n)
    # endregion
    AccuracyList=[]
    xValue=[1]
    accuracy=linkPrediction(embedding,np.array(trainData),trainLabels,np.array(testData),testLabels)
    AccuracyList.append(accuracy)
    #print("Accuracy ",accuracy)
    # region Online Increment
    # STEP 3: over different snapshots of edges, dynamically updating embeddings of nodes and conduct
    #         online anomaly detection for edges, visualize the anomaly score of each snapshot
    snapshotNum = 1
    while (netwalk.hasNext()):
        # region Include next walks dynamically and find embedding
        snapshot_data = netwalk.nextOnehotWalks()
        embedding = getEmbedding(embModel, snapshot_data, n)
        accuracy = linkPrediction(embedding, np.array(trainData), trainLabels, np.array(testData), testLabels)
        AccuracyList.append(accuracy)
        #print("Accuracy ", accuracy)
        snapshotNum += 1
        xValue.append(snapshotNum)
    accuracy = linkPrediction(embedding, np.array(trainData), trainLabels, np.array(testData), testLabels)
    print("Final Accuracy ", accuracy)
    # scores, auc, n0, c0, res, ab_score = anomaly_detection_stream(embedding, train, test_piece, k, alfa, n0, c0)
    # print('Final auc of anomaly detection at snapshot %d: %f' % (snapshotNum, auc))
    # print('Final anomaly score at snapshot %d: %f' % (snapshotNum, ab_score))
    plt.plot(xValue, AccuracyList)
    plt.yticks(np.arange(0, 1, .1))
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
    init_percent = 0.6
    #datasetname=sys.argv[1]
    #datasetname = 'karate'
    datasetname = 'dolphin'
    #
    # datasetname = 'cora'
    #
    # datasetname = 'citeseer'
    #
    #datasetname = 'toy'
    number_walks = 20
    output = './tmp/embedding_' + datasetname + '.txt'
    if (datasetname == "karate"):
        input = '../data/karate.edges'
        snap = 10
        representation_size = 32
    elif (datasetname == "toy"):
        input = '../data/toy2.edges'
        snap = 2
        representation_size = 8
    elif (datasetname == "dolphin"):
        input = '../data/dolphins.mtx'
        snap = 10
        representation_size = 32
    elif (datasetname == "cora"):
        input = '../data/cora.edgelist'
        snap = 500
        representation_size = 128
    elif (datasetname == "citeseer"):
        input = '../data/citeseer.edgelist'
        snap = 500
        representation_size = 128
    walk_length = 3
    # endregion

    static_process(representation_size, walk_length, input, number_walks, init_percent, snap, output, datasetname)


if __name__ == "__main__":
    main()
