import random

class Perceptron:
    def __init__(self, fit_intercept=True, max_iter=1000, l_rate=1.0,
                tol=0.003, shuffle=True, verbose=False):
        # weight and bias
        self.w = None
        self.b = None

        # no of iterations and no of class lables
        self.n_iter = 0
        self.classes = None
        
        # whether to set b == 0 or not(for this it is fixed)
        self.fit_intercept = fit_intercept

        # maximum epochs and learning rate
        self.max_iter = max_iter
        self.l_rate = l_rate
        self.tol = tol
        self.save_wt = False

        # shuffle data after every epoch
        self.shuffle = shuffle
        self.verbose = verbose

        print("Perceptron Model Parameters = (fit_intercept=", self.fit_intercept, ", max_iter=", self.max_iter , ", learning_rate=", self.l_rate,
                "tolerence=", self.tol, ", shuffle=", self.shuffle, ", verbose=", self.verbose, ")\n")

    def __dot(self, w, x):
        # computes __dot product of w and x
        return sum([w[i]*x[i] for i in range(len(x))])

    def __display_weight_binary(self, w=None, b=None):
        # prints the w and b for binary classes
        if w and b:
            print("W = ", ["{0:0.2f}".format(i) for i in  w])
            print("B = ", "{0:0.2f}".format(b)) 
        else:
            print("W = ", ["{0:0.2f}".format(i) for i in  self.w])
            print("B = ", "{0:0.2f}".format(self.b)) 
            
    def __display_weight_multiclass(self, w=None, b=None):
        # prints the w and b for multiclass

        if w and b:
            print("W = ", [["{0:0.2f}".format(i) for i in j] for j in w])
            print("B = ", ["{0:0.2f}".format(i) for i in b])
        else:
            print("W = ", [["{0:0.2f}".format(i) for i in j] for j in self.w])
            print("B = ", ["{0:0.2f}".format(i) for i in self.b])
        
    def __predict_binary(self, x):
        # compute the value of dot(w, x)+b 
        return 1.0 if self.__dot(self.w, x)+self.b>=0.0 else 0.0     
        
    def __predict_multiclass(self, x):
        mat = []
        for i in range(len(self.w)):
            mat.append(self.__dot(self.w[i], x)+self.b[i])
        
        # returns the max dot product class assuming the classes are from 0 to c-1
        return mat.index(max(mat))
               
    def __fit_binary(self, X, Y):
        # saves a copy of weights for each epoch
        wt_matrix = []
        self.data = list(zip(X, Y))
        
        # Initialise the weight and bias randomly
        self.w = [random.random() for i in range(len(X[0]))]
        self.b = random.random()
        
        # set the loss to be maximum as initial value
        self.min_loss = len(X)
        
        # save the weight and bias that gives the minimum loss(no of misclassified points)
        self.checkw = self.w
        self.checkb = self.b
        self.loss = 0
        
        if self.verbose:
            print("Initialised W and B are: ")
            self.__display_weight_binary(self.w, self.b)
            print("Min Loss Inialisd :", self.min_loss)
            print("Value Initialised. Starting the epochs...\n")

        for epoch in range(self.max_iter):
            
            # shuffle the input randomly
            if self.shuffle:
                random.shuffle(self.data)
                
            self.loss = 0    
            if self.verbose:
                print("Epoch = ", epoch, "\n\n") 

            for x, y in self.data:
                y_pred = self.__predict_binary(x)
                
                if y!= y_pred:
                    error = y - y_pred
                    self.loss += 1
                    if self.verbose:
                        print("Found Mismatch! Old values are: ")
                        self.__display_weight_binary(self.w, self.b)
                        print("Updating the values: Adding W", y, " and Subtracting W", y_pred)
                    
                    # update bias and weight
                    self.b = self.b + self.l_rate * error
                    for i in range(len(x)):
                        self.w[i] = self.w[i] + self.l_rate * error * x[i]
            
            if self.verbose: 
                print("Loss Values ", self.loss, self.min_loss) 

            # keeps track of min loss and corresponding weight and bias
            if self.loss < self.min_loss:
                self.min_loss = self.loss
                self.checkw = self.w
                self.checkb = self.b
                self.n_iter = epoch

                if self.verbose:
                    print("Loss updated. Loss = ", self.loss)
                    print("Updated values of W and B are: \n")
                    self.__display_weight_binary(self.w, self.b)
                    print("Check pointed values of W and B are: \n")
                    self.__display_weight_binary(self.checkw, self.checkb)
            if self.loss == 0:
                if self.verbose:
                    print("Early Stopping at epoch:", epoch)
                break
            if self.save_wt:
                wt_matrix.append(self.w)
        
        # restore the values of optimal weight and bias
        self.w = self.checkw
        self.b = self.checkb
        
        
        return wt_matrix
  
    def __fit_multiclass(self, X, Y):

        wt_matrix = []        
        self.data = list(zip(X, Y))

        # Initialise the weight and bias randomly
        self.w = [[random.random() for i in range(len(X[0]))] for j in range(self.classes)]
        self.b = [random.random() for i in range(self.classes)]


            
        # set the loss to be maximum as initial value
        self.min_loss = len(X)
        
        # save the weight and bias that gives the minimum loss(no of misclassified points)
        self.checkw = self.w
        self.checkb = self.b
        self.loss = 0

        if self.verbose:
            print("Initialised W and B are: ")
            self.__display_weight_multiclass(self.w, self.b)
            print("Min Loss Inialisd :", self.min_loss)
            print("Value Initialised. Starting the epochs...\n")

        for epoch in range(self.max_iter):
            # shuffle the input randomly
            if self.shuffle:
                random.shuffle(self.data)
            
            self.loss = 0  
            if self.verbose:
                print("Epoch = ", epoch, "\n\n") 
            for x, y in self.data:
                y_pred = self.__predict_multiclass(x)

                if y != y_pred:
                    self.loss += 1

                    # if self.verbose:
                        # print("Found Mismatch! Old values are: ")
                        # self.__display_weight_multiclass(self.w, self.b)
                        # print("Updating the values: Adding W", y, " and Subtracting W", y_pred)
                    # update bias and weight
                    
                    self.b[y_pred] -= self.l_rate
                    self.b[y] += self.l_rate
                    for i in range(len(x)):
                        self.w[y_pred][i] -= self.l_rate * x[i]
                        self.w[y][i] += self.l_rate * x[i]

            if self.verbose: 
                print("Loss Values ", self.loss, self.min_loss) 
            # keeps track of min loss and corresponding weight and bias
            if self.loss < self.min_loss:
                self.min_loss = self.loss
                self.checkw = self.w
                self.checkb = self.b
                self.n_iter = epoch
                if self.verbose:
                    print("Loss updated. Loss = ", self.loss)
                    print("Updated values of W and B are: \n")
                    self.__display_weight_multiclass(self.w, self.b)
                    print("Check pointed values of W and B are: \n")
                    self.__display_weight_multiclass(self.checkw, self.checkb)
            if self.loss == 0:
                if self.verbose:
                    print("Early Stopping at epoch:", epoch)
                break
            if self.save_wt:
                wt_matrix.append(self.w)
            
            
        # restore the values of optimal weight and bias
        self.w = self.checkw
        self.b = self.checkb
        
        return wt_matrix

    def fit(self, X, Y, ret_wt_matrix=False):
        # fits the perceptron model
        
        self.save_wt = ret_wt_matrix
        
        # get the number of classes so that appropriate algo can be applied
        self.classes = len(set(Y))
    
        if self.classes == 2:
            print("Binary classification...\n")
            wt_matrix = self.__fit_binary(X, Y)
        else:
            print("Multiclass classification...\n")
            wt_matrix = self.__fit_multiclass(X, Y)

        # whether to return the saved weight values
        if self.save_wt:
            return wt_matrix
            
    def predict(self, X):
        # predicts the class for a single input matrix X
        y_pred = []
        if self.classes == 2:
            for x in X:
                y_pred.append(self.__predict_binary(x))
        else:
            for x in X:
                y_pred.append(self.__predict_multiclass(x))
        return y_pred
    
    def accuracy(self, Y, Y_pred):
        # Computes the accuracy score
        count = 0
        for y, y_pred in zip(Y, Y_pred):
            if y == y_pred:
                count += 1
        return count/len(Y)

    def display_weight(self, w=None, b=None):
        if self.classes == 2:
            self.__display_weight_binary(w, b)
        else:
            self.__display_weight_multiclass(w, b)

    