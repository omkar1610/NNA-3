from mlp import FF_multiclass
import numpy as np

from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_val, Y_val) = mnist.load_data()
X_val = X_val.reshape(10000, 784)
X_train = X_train.reshape(60000, 784)


mod = FF_multiclass(layers=[X_train.shape[1], 256, 128, 10], verbose=0)
Y_train, Y_val = mod.onehot(Y_train), mod.onehot(Y_val)

mod.fit(X_train, Y_train,
        epochs=2, learning_rate=1, momentum=0.9,
        mode="batch",
        display_loss=0,
        compute_val_loss=True, X_val=X_val, Y_val=Y_val)

mod.print_accuracy(X_train, Y_train, X_val, Y_val)
print('Best Loss ', mod.best_loss)
