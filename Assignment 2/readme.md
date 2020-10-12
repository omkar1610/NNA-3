# Run the code:

- Extract everything to a folder and run the commands mentioned below for the result.
- `mlp.py` is the class and should be imported to use the model. This depends on `numpy` library for basic vector operations, `matplotlib` for plotting and `sklearn.metrics` for log loss computation.

# MLP Class:

- This model uses deep feed forward neural networks with sigmoid activation in each neuron and softmax for the output layer.
- Log loss is used as the loss function.
  $L = \sum_{i=1}^{n} -\log(\hat{y_i}[l])$. 1. $n$ = No of inputs 2. $l$ = True label of the $i$th input 3. $\hat{y_i}$ = Softmax output for the ith input.

## `model = MLP(layers, verbose)`

- **layers** = `list of integers`: list of neurons in each layers. The first and the last element should be no of features and no of classes respectively.
- **verbosity** = `bool`: verbosity toggle.

## `model.fit(X, Y, epochs=1, learning_rate=0.05, momentum=None, display_loss=False, compute_val_loss=False, X_val=None, Y_val=None, mode='batch', batch_size=32)`

- **X** = `numpy.ndarray of inputs`, $m\times n$: m=No of inputs, n=No of features.
- **Y** = `numpy.ndarray of outputs`, $m\times c$: m=No of inputs, c = no of classes, the list is one hot encoding.
- **learning_rate** = $\eta$, default = 0.05
- **momentum** = $\gamma$, default = None
- **mode** = `'batch'` or `'sgd'` or `'mini_batch'`, default = 'batch'
- **batch_size** = `integer`, available only if `mode = 'mini_batch'`, default = `32`
- **display_loss** = `bool`, display dynamic plot of loss value after each epoch, default = `False`
- **compute_val_loss** = `bool`, compute validation loss and accuracy after each epoch, default = `False`
- **x_val** = if `compute_val_loss`=True, the validation data, constraints same as X
- **y_val** = if `compute_val_loss`=True, the validation data, constraints same as Y

## `model.predict(X)`

- **X** = `numpy.ndarray of inputs`, $m\times n$: m=No of inputs, n=No of features.
- returns the softmax output, size=$m\times c$

## `model.compute_accuracy(y, y_pred)`

- computes the accuracy score.

# Results on different Datasets

|         Datasets          |         IRIS         |         DIGITS         |         MNIST         |
| :-----------------------: | :------------------: | :--------------------: | :-------------------: |
|        Code To Run        | **`python iris.py`** | **`python digits.py`** | **`python mnist.py`** |
| (Inputs, Features, Class) |     (150, 4, 3)      |     (1797, 64, 10)     |   (70000, 784, 10)    |
|          Epochs           |          80          |          200           |          200          |
|       Learning Rate       |          1           |           1            |           1           |
|         Momentum          |         0.9          |          0.9           |          0.9          |
|    Loss: (Train, Test)    |   (0.0546, 0.0425)   |    (0.0483, 0.1234)    |   (0.2871, 0.2951)    |
|  Accuracy: (Train, Test)  |   (0.9833, 0.9667)   |    (0.9916, 0.9639)    |    (0.919, 0.9156)    |

## PS:

- The model was taking a higher amount to train for the MNIST dataset containing **60000** input values.
- I got an accuracy of ~91% with 200 epochs and this could surely be improved with higher epochs.
- To change the parameters such as architecture of the model etc, edit the corresponding value in the source file of the corresponding file.
