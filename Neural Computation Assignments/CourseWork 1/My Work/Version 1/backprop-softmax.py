
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils

# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    return np.divide(1, 1+np.e**(-x))
def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))
def relu(x):
    return np.maximum(x, 0)
def relu_d(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

       
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,20,20,20,10]):

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Number of layers in the network
        self.L = len(network_shape)

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape]
        self.db            = [np.zeros(m) for m in network_shape]
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape]
        self.z             = [np.zeros(m) for m in network_shape]
        self.delta         = [np.zeros(m) for m in network_shape]
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
        self.nabla_C_out   = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi           = relu
        self.phi_d         = relu_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]
                
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """
        self.a[0] = x - 0.5      # Center the input values between [-0.5,0.5]
        # TODO
        for i in range(1, self.L):
            self.z[i] = np.dot(self.w[i], self.a[i-1]) + self.b[i]
            self.a[i] = self.phi(self.z[i])
        self.a[self.L-1] = self.softmax(self.a[self.L-1])
        
        return(self.a[self.L-1])



    def softmax(self, z):
        # TODO
        e_z = np.e**z
        Q = np.sum(np.e**z)
        P = np.divide(e_z, Q)
        return P

    def loss(self, pred, y):
        # TODO
        correct_class = np.argmax(y)
        return -np.log(pred[correct_class])
    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """
        # TODO
        p_j = self.a[self.L-1]
        # print("y: ", y)
        y_class = np.argmax(y)
        # print("y_class: ", y_class)
        # print("a[L-1]: ", self.a[self.L-1])
        # print("p_j: ", p_j)

        # pred_class = np.argmax(self.a[self.L-1])
        # print("pred_class: ", pred_class)

        # if(y_class==pred_class):
        #     self.delta[self.L-1] = p_j - 1
        # else:
        #     self.delta[self.L-1] = p_j - 0
        # print("delta[L-1]: ", self.delta[self.L-1])

        self.delta[self.L-1] = p_j - 0
        self.delta[self.L-1][y_class] = p_j[y_class] - 1
        # print("delta[L-1]", self.delta[self.L-1])

        for i in range(self.L-2, 0, -1):
            self.delta[i] = self.phi_d(self.z[i]) * np.dot(self.w[i+1].T, self.delta[i+1])
        # #     # for j in range(len(self.z[i])):
        # #         # self.delta[i][j] = self.phi_d(self.z[i][j]) * np.sum(self.w[i+1]) * np.sum(self.delta[i+1])
        # #         # self.delta[i] = self.phi_d(self.z[i]) * np.sum(self.w[i+1]) * np.sum(self.delta[i+1])
        #     # print("11111: ", np.dot(self.phi_d(self.z[i]), self.w[i+1].T * self.delta[i+1]))
        #         # print(j)
        # # np.dot(self.w[self.L-5].T, self.delta[self.L-5])
        # print(self.delta[0])

        for i in range(1,self.L-1):
            # self.dw[i] = np.dot(np.mat(self.delta[i]).T, np.mat(self.a[i-1]))
            self.dw[i] = np.outer(self.delta[i], np.mat(self.a[i-1]))
            self.db[i] = self.delta[i]

        # print(np.shape(self.a))

    # Return predicted image class for input x
    def predict(self, x):
        return np.argmax(self.forward(x))

    # # Return predicted percentage for class j
    def predict_pct(self, j):
        return self.a[self.L-1][j]
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    
    def sgd(self,
            batch_size=50,
            epsilon=0.01,
            epochs=1000):

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        
        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N/batch_size)

        # Variables to keep track of statistics
        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True
        
        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):

            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)
            
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append( self.evaluate(self.testX, self.testY, 1000) )
            train_acc_log.append( self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            for k in range(num_batches):
                
                # Reset buffer containing updates
                # TODO
                for i in range(self.L):
                    self.db[i].fill(0)
                    self.delta[i].fill(0)
                    self.dw[i].fill(0)
                    self.a[i].fill(0)
                    self.z[i].fill(0)
                
                # Mini-batch loop
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    # TODO
                    self.forward(x)
                    # Compute gradients
                    # TODO
                    self.backward(x, y)

                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                                    
                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                    self.w[l] = self.w[l] - epsilon * self.dw[l]# TODO
                    self.b[l] = self.b[l] - epsilon * self.db[l]# TODO
                
                # Update logs
                loss_log.append( batch_loss / batch_size )
                batch_loss = 0

                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)

                # # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self,show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)


# Start training with default parameters.

def main():
    bp = BackPropagation()
    bp.sgd()

    # Task 1 Test - softmax
    # z = np.array([0,1,2,3,4,5,6,7,8,9])
    # print(bp.softmax(z))

    # Task 2 Test - forward
    # print(bp.forward(bp.trainX[0]))

    #Task 3 Test(1) - loss
    # print("a[L-1]: ", bp.forward(bp.trainX[0]))
    # print("one hot encode for correct class", bp.trainY[0])
    # print("loss: ", bp.loss(bp.a[bp.L-1], bp.trainY[0]))
    #Task 3 Test(2) - loss
    # testX = np.array([0.09371258, 0.10556965, 0.09432195, 0.10503915, 0.12660278, 0.10768817, 0.08212865, 0.10272175, 0.07511968, 0.10709562])
    # testY = np.array([0,0,0,0,1,0,0,0,0,0])
    # print("a[L-1]: ", testX)
    # print("one hot encode for correct class", testY)
    # print("loss: ", bp.loss(testX, testY))

    # Task 4 Test - backward
    # bp.forward(bp.trainX[0])
    # bp.backward(bp.trainX[0], bp.trainY[0])
    # print("bp.dw[1]: ", bp.dw[1])

    #Task 5 Test - predict
    # print("predict class", bp.predict(bp.trainX[0]))
    # print("a[L-1]", bp.a[bp.L-1])

    #Task 5 Test - predict_pct
    # bp.forward(bp.trainX[0])
    # print("a[L-1]", bp.a[bp.L-1])
    # a = 6
    # print("predict class ", a, ": ", bp.predict_pct(a))


if __name__ == "__main__":
    main()
    