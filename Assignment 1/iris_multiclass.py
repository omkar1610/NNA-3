from perceptron import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    X = load_iris().data 
    Y = load_iris().target
    
    # split the data to train test with 80:20 ratio
    X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.2)
    
    # perceptron model
    model = Perceptron(shuffle=1, verbose=0)

    # converting to  list as the perceptron model doesn't recognize np arrays.
    mat = model.fit(list(X_tr), list(Y_tr))
    
    print("The Weight and Bias :")
    model.display_weight()
    print("\nAccuracy : ")
    print("Train Accuracy : ", model.accuracy(list(Y_tr), model.predict(list(X_tr))))
    print("Test Accuracy : ", model.accuracy(list(Y_ts), model.predict(list(X_ts))))
    print("No of iterations : ", model.n_iter)