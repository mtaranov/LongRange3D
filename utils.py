import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats.mstats import mquantiles
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def concat_motifs_old(X1_path, X2_path):
    X1=np.load(X1_path)
    X2=np.load(X2_path)
    if X1.shape != X2.shape:
        raise ValueError('Nodes are of different dimensions!')
    _, num_features = X1.shape    
    X1_new = X1.reshape(len(X1), 1, num_features, 1)
    X2_new = X2.reshape(len(X2), 1, num_features, 1)
    return np.concatenate((X1_new, X2_new), axis=3) 

def concat_motifs(X_path):
    if len(X_path)==2:
        X1=np.load(X_path[0])
        X2=np.load(X_path[1])
        if X1.shape != X2.shape:
            raise ValueError('Nodes are of different dimensions!')
        _, num_features = X1.shape
        X1_new = X1.reshape(len(X1), 1, num_features, 1)
        X2_new = X2.reshape(len(X2), 1, num_features, 1)
        return np.concatenate((X1_new, X2_new), axis=3)
    elif len(X_path)==3:
        X1=np.load(X_path[0])
        X2=np.load(X_path[1])
        X3=np.load(X_path[2])
        if X1.shape != X2.shape or X1.shape != X3.shape:
            raise ValueError('Nodes are of different dimensions!')
        _, num_features = X1.shape
        X1_new = X1.reshape(len(X1), 1, num_features, 1)
        X2_new = X2.reshape(len(X2), 1, num_features, 1)
        X3_new = X3.reshape(len(X2), 1, num_features, 1)
        return np.concatenate((np.concatenate((X1_new, X2_new), axis=3), X3_new), axis=3)



def get_features(X_path):
    # load data
    X = np.load(X_path)
    # reshape data
    _, num_features, num_nodes = X.shape
    return X.reshape(len(X), 1, num_features, num_nodes)


def get_labels(y_path):
    # load data
    y = np.load(y_path)
    # reshape data
    return y.reshape((len(y)), 1).astype(bool)

def subsample_data(X, y, imbalance_ratio=10):
    neg_indxs = np.where(y==False)[0]
    pos_indxs = np.where(y==True)[0]
    num_negatives = len(pos_indxs)*imbalance_ratio
    y_subsampled = np.array([y[i] for i in np.concatenate((pos_indxs, neg_indxs[:num_negatives]))])
    X_subsampled = np.array([X[i] for i in np.concatenate((pos_indxs, neg_indxs[:num_negatives]))])

    return (X_subsampled, y_subsampled)

def normalize_features(X_train, X_valid, X_test, normalizer=StandardScaler):
    # fit normalizer
    normalizer = normalizer().fit(np.concatenate((X_train[:, 0, :, 0], X_train[:, 0, :, 1]), axis=0))
    # transform features
    X_train_new=copy.copy(X_train)
    X_valid_new=copy.copy(X_valid)
    X_test_new=copy.copy(X_test)
    X_train_new[:, 0, :, 0] = normalizer.transform(X_train[:, 0, :, 0])
    X_train_new[:, 0, :, 1] = normalizer.transform(X_train[:, 0, :, 1])
    X_valid_new[:, 0, :, 0] = normalizer.transform(X_valid[:, 0, :, 0])
    X_valid_new[:, 0, :, 1] = normalizer.transform(X_valid[:, 0, :, 1])
    X_test_new[:, 0, :, 0] = normalizer.transform(X_test[:, 0, :, 0])
    X_test_new[:, 0, :, 1] = normalizer.transform(X_test[:, 0, :, 1])

    return (X_train_new, X_valid_new, X_test_new)

def normalize_features_sampl_by_f(X_train, X_valid, X_test, normalizer=StandardScaler):
    # fit normalizer
    normalizer = normalizer().fit(X_train)
    # transform features
    X_train_new=copy.copy(X_train)
    X_valid_new=copy.copy(X_valid)
    X_test_new=copy.copy(X_test)
    X_train_new = normalizer.transform(X_train)
    X_valid_new = normalizer.transform(X_valid)
    X_test_new = normalizer.transform(X_test)

    return (X_train_new, X_valid_new, X_test_new)

# builds adjacency matrix 
def reconstruct_2d(indx, labels, matrixSize):

    # Initialize matrix (promoter x promoter)
    matrix=np.ones((matrixSize, matrixSize)) #  number of promoters in chr 1
    matrix=-1*matrix
    # Fill (promoter x promoter) matrix with q-values of promoter-promoter interaction
    k=0
    for i in indx:
        matrix[int(i[0]), int(i[1])]=labels[k]
        k+=1
    print "Some tests on adjacency matrix:"
    # 1. Check if the matrix is symmetric:
    #if (matrix.transpose() == matrix).all() == True:
        #print "Adjacency matrix is symmetric"

    return matrix

def printMatrix(Matrix, ylabel, QuantileValue, LowerUpperLimit, title=''):
    #vmaxLim=mquantiles(Matrix,[0.99])[0]
    Lim=mquantiles(Matrix,[QuantileValue])[0]
    print Matrix.max()
    print np.shape(Matrix)
    print "Limit:", Lim
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    if LowerUpperLimit == 'lower':
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmin=Lim)
    else:
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmax=Lim) # cmap="RdYlBu_r"


    ax.axhline(-0.5, color="#000000", linewidth=1, linestyle="--")
    ax.axvline(-0.5, color="#000000", linewidth=1, linestyle="--")

    cb = fig.colorbar(m)
    cb.set_label(ylabel)

    ax.set_ylim((-0.5, len(Matrix) - 0.5))
    ax.set_xlim((-0.5, len(Matrix) - 0.5))
    
    plt.title(title)
    plt.show()
    return

def zscore(d): return (d-d.mean(0))/d.std(0)

def binarize(matrix, thres=0.5):
    matrix=copy.copy(matrix)
    matrix[matrix <= thres] = 0
    matrix[matrix > thres] = 1    
    return matrix

def binarize_w_unlabeled(matrix, thres):
    matrix2=copy.copy(matrix)
    matrix2[matrix == -1] = -1
    matrix2[matrix >= thres] = 1
    matrix2[np.logical_and(matrix>=0, matrix<thres)] = 0
    return matrix2

def change_scale(matrix):
    matrix2=copy.copy(matrix)
    matrix2[matrix == -1] = 0
    matrix2[matrix == 0] = -1
    return matrix2

def get_2D(indx, preds, labels, NumberOfNodes):
    preds_2d = reconstruct_2d(indx, preds, NumberOfNodes)
    labels_2d = reconstruct_2d(indx, labels, NumberOfNodes)
    return preds_2d, labels_2d

def plot_prediction(preds_2d, labels_2d, zoomIn_window):
    w=zoomIn_window
    subset_labels_2d=labels_2d[[i for i in range(w[0], w[1])]][:, [i for i in range(w[0], w[1])]]    
    #subset_preds_2d=binarize_w_unlabeled(preds_2d, preds_thres)[[i for i in range(w[0], w[1])]][:, [i for i in range(w[0], w[1])]]
    subset_preds_2d=preds_2d[[i for i in range(w[0], w[1])]][:, [i for i in range(w[0], w[1])]]
    printMatrix(subset_labels_2d, '', 1, i, title='labels')
    printMatrix(subset_preds_2d, '', 1, i, title='preds') 
