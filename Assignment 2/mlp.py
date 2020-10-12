import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import tqdm
import random


class FF_multiclass:
    def __init__(self, layers, verbose=False):
        self.verbose = verbose
        self.layers = layers
        self.n_layers = len(self.layers)
        self.params = {}
        self.gradients = {}
        self.update_params = {}
        self.prev_update_params = {}
        self.best_params = {}
        self.best_loss = np.inf
        self.best_train_acc = 0

        for i in range(1, self.n_layers):
            self.params['W'+str(i)] = np.random.randn(self.layers[i-1],
                                                      self.layers[i])/np.sqrt(self.layers[i-1])
            self.params['B'+str(i)] = np.random.randn(1,
                                                      self.layers[i])/np.sqrt(self.layers[i-1])

            self.gradients['W' + str(i)] = 0
            self.gradients['B' + str(i)] = 0

            self.update_params['W' + str(i)] = 0
            self.update_params['B' + str(i)] = 0

            self.prev_update_params['W' + str(i)] = 0
            self.prev_update_params['B' + str(i)] = 0

    def sigmoid(self, X):
        """
        A numerically stable version of the logistic sigmoid function.
        """
        pos_mask = (X >= 0)
        neg_mask = (X < 0)
        z = np.zeros_like(X)
        z[pos_mask] = np.exp(-X[pos_mask])
        z[neg_mask] = np.exp(X[neg_mask])
        top = np.ones_like(X)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

    def softmax(self, X):
        exps = np.exp(X - X.max(axis=1).reshape(-1, 1))
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def grad_sigmoid(self, X):
        return X*(1-X)

    def forward_pass(self, X):

        params = self.params
        if self.verbose:
            print('Forward Pass. Layer: ', end=' ')
        # input
        i = 1
        params['A'+str(i)] = np.matmul(X, params['W'+str(i)]) + \
            params['B'+str(i)]
        params['H'+str(i)] = self.sigmoid(params['A'+str(i)])
        if self.verbose:
            print(i, end=' ')

        # hidden
        for i in range(2, self.n_layers-1):
            params['A'+str(i)] = np.matmul(params['H'+str(i-1)],
                                           params['W'+str(i)]) + params['B'+str(i)]
            params['H'+str(i)] = self.sigmoid(params['A'+str(i)])
            if self.verbose:
                print(i, end=' ')
        # output
        i = self.n_layers-1
        params['A'+str(i)] = np.matmul(params['H'+str(i-1)],
                                       params['W'+str(i)]) + params['B'+str(i)]
        params['H'+str(i)] = self.softmax(params['A'+str(i)])
        if self.verbose:
            print(i, 'Completed.')

        # for prediction
        return params['H'+str(i)]

    def compute_grad(self, X, Y):

        params = self.params
        gradients = self.gradients

        # self.forward_pass(X)
        if self.verbose:
            print('Backward Pass. Layer: ', end=' ')
        # output layer
        i = self.n_layers-1

        # compute derivative depending on loss = cross entropy and output = softmax
        gradients['A'+str(i)] = self.params['H'+str(i)] - Y

        # hidden layers
        for i in range(self.n_layers-1, 1, -1):

            # Now normal delta values computation
            gradients['W'+str(i)] = np.matmul(params['H' +
                                                     str(i-1)].T, gradients['A'+str(i)])
            gradients['B'+str(i)] = np.sum(gradients['A' +
                                                     str(i)], axis=0).reshape(1, -1)

            gradients['H'+str(i-1)] = np.matmul(gradients['A' +
                                                          str(i)], params['W'+str(i)].T)
            gradients['A'+str(i-1)] = np.multiply(gradients['H'+str(i-1)],
                                                  self.grad_sigmoid(params['H'+str(i-1)]))
            if self.verbose:
                print(i, end=' ')

        # update W1 and B1
        i = 1
        # print(gradients['W'+str(i)].shape, X.T.shape, len(gradients['A'+str(,) gradients['A'+str(2)].shape, 'H')
        gradients['W'+str(i)] = np.matmul(X.T, gradients['A'+str(i)])
        gradients['B'+str(i)] = np.sum(gradients['A' +
                                                 str(i)], axis=0).reshape(1, -1)
        if self.verbose:
            print(i, 'Completed.')

    def fit(self, X, Y,
            epochs=1, learning_rate=0.05, momentum=None,
            display_loss=False,
            compute_val_loss=False, X_val=None, Y_val=None,
            mode='batch', batch_size=32):

        # one hot encoding of Y
        # Y = self.onehot(Y)

        self.m = X.shape[0]
        self.display_loss = display_loss
        self.loss = {}

        self.learning_rate = learning_rate
        self.momentum = momentum

        if self.display_loss:
            # for plotting
            xdata = []
            ydata = []
            plt.show()
            axes = plt.gca()
            axes.set_xlim(0, epochs)
            axes.set_ylim(0, 1)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            line, = axes.plot(xdata, ydata, '-o', markersize=5)

        X_normalised = self.normalize_data(X, fit=True)
        if compute_val_loss:
            X_val_normalised = self.normalize_data(X_val)
            # Y_val = self.onehot(Y_val)

        if mode == 'batch':
            batch_size = X.shape[0]
        elif mode == 'sgd':
            batch_size = 1
        elif mode == 'mini_batch':
            batch_size = 32

        for i in range(epochs):
            print('Epoch ', i, end=': ')

            for batch in list(range(0, X.shape[0], batch_size)):
                if self.verbose:
                    print('Batch: ', batch//batch_size)

                X_, Y_ = X_normalised[i:i+batch_size], Y[i:i+batch_size]
                self.y_pred = self.forward_pass(X_)
                self.compute_grad(X_, Y_)
                self.update_weights(batch_size, X_, Y_)

            if self.verbose:
                print('Done.')

            y_pred = self.forward_pass(X_normalised)
            self.loss[i] = log_loss(
                np.argmax(Y, axis=1), np.array(y_pred).squeeze())
            acc = self.compute_accuracy(Y, y_pred)

            # keep track of best params
            if self.loss[i] < self.best_loss:
                self.best_loss = self.loss[i]
                self.best_train_acc = acc
                self.best_params = self.params.copy()

            if compute_val_loss:
                y_pred_val = self.forward_pass(X_val_normalised)
                loss_val = log_loss(
                    np.argmax(Y_val, axis=1), np.array(y_pred_val).squeeze())
                acc_val = self.compute_accuracy(Y_val, y_pred_val)
                print('Train loss: ', round(self.loss[i], 4),
                      ', Accuracy: ', round(acc, 4),
                      ', Val loss: ', round(loss_val, 4),
                      ', Accuracy: ', round(acc_val, 4))
            else:
                print('Train loss: ', round(
                    self.loss[i], 4), ', Accuracy: ', round(acc, 4))

            if self.display_loss:
                # Plot Dynamic
                xdata.append(i)
                ydata.append(self.loss[i])
                line.set_xdata(xdata)
                line.set_ydata(ydata)
                axes.set_ylim(min(ydata), max(ydata))
                plt.draw()
                plt.pause(1e-17)
        # restore the best params
        self.params = self.best_params

        if self.display_loss:
            # add this if you don't want the window to disappear at the end
            plt.show()

    def update_weights(self, m, X, Y):
        # this goes inside each epoch
        if self.momentum is None:
            for j in range(1, self.n_layers):
                # update weight and bias
                self.params['W'+str(j)] -= self.learning_rate * \
                    (self.gradients['W'+str(j)]/m)
                self.params['B'+str(j)] -= self.learning_rate * \
                    (self.gradients['B'+str(j)]/m)

        else:
            for j in range(1, self.n_layers):
                # save the prev updates
                self.update_params["W"+str(j)] = self.momentum * self.update_params["W" +
                                                                                    str(j)] + self.learning_rate * (self.gradients["W"+str(j)]/m)
                self.update_params["B"+str(j)] = self.momentum * self.update_params["B" +
                                                                                    str(j)] + self.learning_rate * (self.gradients["B"+str(j)]/m)
                # update weights
                self.params["W"+str(j)] -= self.update_params["W"+str(j)]
                self.params["B"+str(j)] -= self.update_params["B"+str(j)]

    def predict(self, X):

        y_pred = self.forward_pass(self.normalize_data(X))
        return np.array(y_pred).squeeze()

    def compute_accuracy(self, y, y_pred):
        y, y_pred = np.argmax(y, axis=1), np.argmax(y_pred, axis=1)
        return 1 - np.count_nonzero(y-y_pred)/y.shape[0]

    def print_accuracy(self, X_train, Y_train, X_val, Y_val):
        accuracy_train = self.compute_accuracy(self.predict(X_train), Y_train)
        print("Training accuracy", round(accuracy_train, 4))

        accuracy_val = self.compute_accuracy(self.predict(X_val), Y_val)
        print("Validation accuracy", round(accuracy_val, 4))

    def normalize_data(self, X, fit=False):
        if fit:
            self.max_norm, self.min_norm = X.max(
                axis=0).reshape(1, -1), X.min(axis=0).reshape(1, -1)
            self.diff_norm = self.max_norm - self.min_norm
            self.diff_norm[self.diff_norm == 0] = 1
        return (X-self.min_norm)/(self.diff_norm)
        # return X

    def onehot(self, y):
        n_classes = len(set(y))

        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)] for i in range(len(y))])
