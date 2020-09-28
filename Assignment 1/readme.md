## Steps to run the files: (Command : `python <file_name.py>`)

### Extract all the files to a single folder.
### `python iris_binary.py` - 
IRIS data(combining class 1 and 2) is used for the binary classification purpose.
Train and Test Accuracy = 1 and 1
### `python iris_multiclass.py` - 
IRIS data is used for the multi class classification purpose.
Train and Test Accuracy = .975 and 1
### `python normal_data_binary.py` - 
The randomly generated normally distributed 2D points are used for binary classification.
Train and Test Accuracy = 1 and 1


## Declarations:

* The `perceptron.py` contains the perceptron class which will be imported in all other files.
* This class is not using any other libraries for implementing the perceptron class
* `sklearn` is only used to load the dataset, split the dataset and nothing else.
* `numpy` is only used to generate the 2D normally distributed data and nothing else.
* `pandas` and `matplotlib` is used only for plotting in the 4th problem.
* Accuracy score has been used as the evaluation metric.
* Since completely random numbers are used, the results are not same for every execution of the code. The numbers here are the max train and test accuracy that I got.


## Perceptron model:
### Import the class-
* `from perceptron import Perceptron`
    
### Create the model-
* `model = Perceptron(max_iter=1000, l_rate=1.0, shuffle=True, verbose=False)`
    
    max_iter = Specify maximum iterations to run
    l_rate = Learning Rate
    shuffle = Whether to shuffle data after every epoch
    verbose = Display intermediate values
    
### Train the model-
* `model.fit(X, Y)`
    
    X = list of lists(M x N list), M = no of data points, N = No of features
    Y = list of class labels(M x 1 list)
    
### Predict the lables-
* `model.predict(X)`
            
     X = list of list(M x N list, ), M = no of data points, N = No of features
    
### Compute accuracy score-
* `model.accuracy(Y, Y_pred)`
           
    Y = List of original class labels (M x 1)
    Y_pred = List of predicted class labels (M x 1)
    
### Display the weight and bias-
* `model.display_weight()`


