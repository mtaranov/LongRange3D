import numpy as np
import os
import sys
from abc import abstractmethod, ABCMeta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import optimizers
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape, 
)
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.regularizers import l1

from metrics import ClassificationResult 
#from deeplift import keras_conversion as kc
#from deeplift.blobs import MxtsMode

from sklearn.svm import SVC as scikit_SVC
from sklearn.tree import DecisionTreeClassifier as scikit_DecisionTree
from sklearn.ensemble import RandomForestClassifier

class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **hyperparameters):
        pass

    @abstractmethod
    def train(self, X, y, validation_data):
        pass
    @abstractmethod
    def predict(self, X):
        pass

    def test(self, X, y):
        return ClassificationResult(y, self.predict(X))

    def score(self, X, y, metric):
        return self.test(X, y)[metric]

class LongRangeDNN(Model):

    class PrintMetrics(Callback):

        def __init__(self, validation_data, sequence_DNN):
            self.X_valid, self.y_valid = validation_data
            self.sequence_DNN = sequence_DNN

        def on_epoch_end(self, epoch, logs={}):
            print('Epoch {}: validation loss: {:.3f}\n{}\n'.format(
                epoch,
                logs['val_loss'],
                self.sequence_DNN.test(self.X_valid, self.y_valid)))

    class LossHistory(Callback):

        def __init__(self, X_train, y_train, validation_data, sequence_DNN):
            self.X_train = X_train
            self.y_train = y_train
            self.X_valid, self.y_valid = validation_data
            self.sequence_DNN = sequence_DNN
            self.train_losses = []
            self.valid_losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(self.sequence_DNN.model.evaluate(
                self.X_train, self.y_train, verbose=False))
            self.valid_losses.append(self.sequence_DNN.model.evaluate(
                self.X_valid, self.y_valid, verbose=False))


    def __init__(self, num_features=11, num_nodes=2, use_deep_CNN=False,
                  num_tasks=1, num_filters=2000,
                  num_filters_2=2000, 
                  L1=0, dropout=0.4, verbose=2):
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.input_shape = (1, num_features, num_nodes)
        self.num_tasks = num_tasks
        self.verbose = verbose
        self.model = Sequential()
        self.model.add(Convolution2D(
            nb_filter=num_filters, nb_row=num_features,
            nb_col=1, activation='linear',
            init='he_normal', input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        if use_deep_CNN:
            for i in range(3):
                self.model.add(Convolution2D(
                    nb_filter=num_filters_2, nb_row=1,
                    nb_col=1, activation='relu',
                    init='he_normal', W_regularizer=l1(L1)))
                self.model.add(Dropout(0.0))
        self.model.add(Flatten())
        for i in range(4):
            self.model.add(Dense(output_dim=2000, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(dropout))
        #self.model.add(Dropout(0.4))
        self.model.add(Dense(output_dim=self.num_tasks))
        self.model.add(Activation('sigmoid'))
        adam=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=adam, loss='binary_crossentropy')
        #self.model.compile(optimizer='adam', loss='binary_crossentropy')
        #sgd=optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
        #self.model.compile(optimizer=sgd, loss='binary_crossentropy')
        #self.model.compile(optimizer='sgd', loss='binary_crossentropy')
        #self.model.compile(loss='binary_crossentropy', optimizer=sgd)
        self.train_losses = None
        self.valid_losses = None

    def train(self, X, y, validation_data):
        if y.dtype != bool:
            assert len(np.unique(y)) == 2
            y = y.astype(bool)
        multitask = y.shape[1] > 1
        if not multitask:
            num_positives = y.sum()
            num_sequences = len(y)
            num_negatives = num_sequences - num_positives
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=20)]
        if self.verbose >= 1:
            self.callbacks.append(self.PrintMetrics(validation_data, self))
            print('Training model...')
        self.callbacks.append(self.LossHistory(X, y, validation_data, self))
        self.model.fit(
            X, y, batch_size=500, nb_epoch=100,
            validation_data=validation_data,
            class_weight={True: num_sequences / num_positives,
                          False: num_sequences / num_negatives}
            if not multitask else None,
            callbacks=self.callbacks, verbose=self.verbose >= 2)
        self.train_losses = self.callbacks[-1].train_losses
        self.valid_losses = self.callbacks[-1].valid_losses
           
    def predict(self, X):
        return self.model.predict(X, batch_size=128, verbose=False) 

    def save(self, prefix, path):
        arch_fname = path + 'models/' + prefix + '.arch.json'
        #arch_fname = '/users/mtaranov/LongRange3D/models/dnn_CONV_motifs.arch.json'
        weights_fname = path + 'weights/' + prefix + '.weights.h5'
        #weights_fname = '/users/mtaranov/LongRange3D/weights/dnn_CONV_motifs.weights.h5'
        open(arch_fname, 'w').write(self.model.to_json())
        self.model.save_weights(weights_fname, overwrite=True)


    def deeplift(self, X, keras_model_weights, keras_model_json, batch_size=128):
        """
        Returns (num_task, num_samples, 1, num_bases, sequence_length) deeplift score array.
        """
        if sys.version_info[0] != 2:
            raise RuntimeError("DeepLIFT requires Python2!")
        assert len(np.shape(X)) == 4 and np.shape(X)[1] == 1
        from deeplift.conversion import keras_conversion as kc
        from deeplift.blobs import NonlinearMxtsMode

        #load the keras model
        keras_model = kc.load_keras_model(weights=keras_model_weights,
                                  json=keras_model_json)

        # normalize sequence convolution weights
        #kc.mean_normalise_first_conv_layer_weights(self.model, True,None)

        # run deeplift
        deeplift_model = kc.convert_sequential_model(
           self.model, nonlinear_mxts_mode=NonlinearMxtsMode.Gradient)

        # compile scoring function
        target_contribs_func = deeplift_model.get_target_contribs_func(
            find_scores_layer_idx=0, target_layer_idx=-2)

        input_reference_shape = tuple([1] + list(X.shape[1:]))
        return np.asarray(target_contribs_func(task_idx=0, input_data_list=[X],
                                 batch_size=batch_size, progress_update=None,
                                 input_references_list=[np.zeros(input_reference_shape)]))

class LongRangeDNN_FC(LongRangeDNN):
    def __init__(self, num_features=22, use_deep_CNN=False,
                  num_tasks=1, num_filters=(5000, 5000, 5000),
                  #num_tasks=1, num_filters=(1000,1000,1000, 1000,1000,1000,1000,1000,1000,1000,1000, 1000,1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
                  #num_tasks=1, num_filters=(2000,2000,2000, 2000,2000,2000,2000,2000,2000,2000,2000, 2000,2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000),
                  #num_tasks=1, num_filters=(2000,2000,2000, 2000,2000,2000,2000,2000,2000,2000,2000),
                  L1=0.0, dropout=0, verbose=2):
        self.num_features = num_features
        self.input_shape = (1, num_features)
        self.num_tasks = num_tasks
        self.verbose = verbose
        self.model = Sequential()
        for filter_size in num_filters:
            #self.model.add(Dense(filter_size, input_dim=self.num_features,init='he_normal', activation='linear'))
            self.model.add(Dense(filter_size, input_dim=self.num_features,init='he_normal'))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.2))
        self.model.add(Dropout(0.0))
#        if use_deep_CNN:
#            self.model.add(Convolution2D(
#                nb_filter=num_filters_2, nb_row=1,
#                nb_col=1, activation='relu',
#                init='he_normal', W_regularizer=l1(L1)))
#            self.model.add(Dropout(dropout))
#            self.model.add(Convolution2D(
#                nb_filter=num_filters_3, nb_row=1,
#                nb_col=1, activation='relu',
#                init='he_normal', W_regularizer=l1(L1)))
#            self.model.add(Dropout(dropout))
        self.model.add(Dense(output_dim=self.num_tasks))
        self.model.add(Activation('sigmoid'))
        adam=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=adam, loss='binary_crossentropy')
        #self.model.compile(optimizer='adam', loss='binary_crossentropy')
        #sgd=optimizers.SGD(lr=0.008, momentum=0.0, decay=0.0, nesterov=False)
        #self.model.compile(optimizer=sgd, loss='binary_crossentropy')
        #self.model.compile(optimizer='sgd', loss='binary_crossentropy')
        #self.model.compile(loss='binary_crossentropy', optimizer=sgd)
        self.train_losses = None
        self.valid_losses = None

class DecisionTree(Model):

    def __init__(self):
        self.classifier = scikit_DecisionTree()

    def train(self, X, y, validation_data=None):
        self.classifier.fit(X, y)

    def predict(self, X):
        predictions = np.asarray(self.classifier.predict_proba(X))[..., 1]
        if len(predictions.shape) == 2:  # multitask
            predictions = predictions.T
        else:  # single-task
            predictions = np.expand_dims(predictions, 1)
        return predictions


class RandomForest(DecisionTree):

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)

    def ImportanceSelect(self):
        #return X[:,self.classifier.feature_importances_.argsort()[::-1][:k]]
        return self.classifier.feature_importances_
    

class SVC(Model):

    def __init__(self):
        #self.classifier = scikit_SVC(probability=True, kernel='linear')
        self.classifier = scikit_SVC(probability=True, kernel='rbf')

    def train(self, X, y, validation_data=None):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict_proba(X)[:, 1:]
