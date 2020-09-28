# Using the perceptron model on two linearly separable classes
# Numpy library is used only to generate normal points
# pandas and matplotlib is used for the plotting only.

from perceptron import Perceptron
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def generate_random_points(dim=2, class_count=[50, 60], mean=10, var=2, step=100):
    '''
    This function simply generates normal points of 'mean' and 'variance', and keep on adding a bias of size 'step' for each class.
    The c class lables are [0, 1 , 2, ... , c-1]

    dim : Integer, The dimension of the data
    class_count : Array of integers, List of counts for each class

    Example : 
        dim = 3 i.e each point is of 3 dimension
        class_count = [10, 20, 15] means 10 points of class 0, 20 points of class 1 and 15 points of class 2
    '''
    x = np.random.normal(mean, var, size=[class_count[0],dim])
    y = [0 for i in range(class_count[0])]
    i = 1
    for n in class_count[1:]:
        tmp = np.random.normal(10, 2, size=[n,dim]) + i * step
        i += 1
        x = np.vstack((x, tmp))
        y = np.hstack((y, [i-1 for j in range(n)]))
    return x, y

def plot_2d(x, y, w, b):
    df = pd.DataFrame(x)
    df['y'] = y
    
    plt.scatter(df[0], df[1])

    xa = np.linspace(0, 200, 2)

    ya = (w[0]*xa+b)/(-w[1])
    plt.plot(xa, ya, 'r')

    
    plt.grid(1)
    plt.show()
    

if __name__ == '__main__':
    X, Y = generate_random_points(class_count=[500, 500,])
    print("Random 2D Normally distributed Points Generated...\n")
    
    # split the data to train test with 80:20 ratio
    X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.2)
    
    # perceptron model
    model = Perceptron(shuffle=1, verbose=0)

    # converting to  list as the perceptron model doesn't recognize np arrays.
    mat = model.fit(X_tr, Y_tr)
    
    print("\nThe Weight and Bias :")
    model.display_weight()
    print("\nAccuracy : ")
    print("Train Accuracy : ", model.accuracy(Y_tr, model.predict(X_tr)))
    print("Test Accuracy : ", model.accuracy(Y_ts, model.predict(X_ts)))
    print("No of iterations : ", model.n_iter)
    if len(set(Y)) == 2:
        plot_2d(X, Y, model.w, model.b)
    else:
        print("Not plotting as the data is not 2D")

    