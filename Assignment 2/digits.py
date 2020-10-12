from mlp import FF_multiclass
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

data = load_digits()
X_train, X_val, Y_train, Y_val = train_test_split(
    data['data'], data['target'], test_size=0.20, random_state=42)

mod = FF_multiclass(layers=[X_train.shape[1], 20, 15, 10], verbose=0)
Y_train, Y_val = mod.onehot(Y_train), mod.onehot(Y_val)

mod.fit(X_train, Y_train, epochs=150, learning_rate=1,
        mode="batch", display_loss=1, momentum=0.9, compute_val_loss=True, X_val=X_val, Y_val=Y_val)

mod.print_accuracy(X_train, Y_train, X_val, Y_val)
print('Best Loss ', mod.best_loss)
