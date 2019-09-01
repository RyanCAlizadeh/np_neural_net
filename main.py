import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigGrad(z):
    a = sigmoid(z)
    return a * (1 - a)

def feedForward(theta1, theta2, theta3, X):
    a1 = np.hstack((1, X))
    z2 = a1 @ theta1
    a2 = np.hstack((1, sigmoid(z2)))
    z3 = a2 @ theta2
    a3 = np.hstack((1, sigmoid(z3)))
    z4 = a3 @ theta3
    a4 = sigmoid(z4)

def cost(theta1, theta2, theta3, X, y):
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1 @ theta1
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = a2 @ theta2
    a3 = np.hstack((np.ones((z3.shape[0], 1)), sigmoid(z3)))
    z4 = a3 @ theta3
    a4 = sigmoid(z4)

    c = np.arange(10)
    yvec = (c == np.array([y]).T) * 1
    j = sum(sum(-yvec * np.log(a4) - (1-yvec) * np.log(1-a4))) / m

    return j

def computeGrad(theta1, theta2, theta3, X, y):
    m = X.shape[0]
    c = np.arange(10)
    yvec = (c == np.array([y]).T) * 1
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1 @ theta1
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = a2 @ theta2
    a3 = np.hstack((np.ones((z3.shape[0], 1)), sigmoid(z3)))
    z4 = a3 @ theta3
    a4 = sigmoid(z4)
    
    delta4 = a4 - yvec # (60000, 10)
    delta3 = delta4 @ theta3.T * np.hstack((np.ones((z3.shape[0], 1)), sigGrad(z3))) # (60000, 17)
    delta3 = delta3[:, 1:] # (60000, 16)
    delta2 = delta3 @ theta2.T * np.hstack((np.ones((z2.shape[0], 1)), sigGrad(z2)))
    delta2 = delta2[:, 1:] # (60000, 16)

    # print(delta4.shape) #(60000, 10)
    # print(delta3.shape) #(60000, 16)
    # print(delta2.shape) #(60000, 16)
    # print(a3.shape) #(60000, 17)
    # print(a2.shape) #(60000, 17)
    # print(a1.shape) #(60000, 785)

    t3grad = (a3.T @ delta4) / m
    t2grad = (a2.T @ delta3) / m
    t1grad = (a1.T @ delta2) / m
    
    # print(theta3.shape) # (17, 10)
    # print(theta2.shape) # (17, 16)
    # print(theta1.shape) # (785, 16)
    
    # print(t3grad.shape) # (17, 10)
    # print(t2grad.shape) # (17, 16)
    # print(t1grad.shape) # (785, 16)

    return t1grad, t2grad, t3grad

(xtrain, y), (xtest, ytest) = keras.datasets.mnist.load_data()

num_in = 784
hid_size = 16
out_size = 10
alpha = 0.005

xtrain = xtrain / 256
xtest = xtest / 256
X = np.zeros((60000, 784))


for i in range(len(xtrain)):
    X[i] = xtrain[i].flatten()

m = X.shape[0]

theta1 = np.random.rand(num_in + 1, hid_size) - 0.5
theta2 = np.random.rand(hid_size + 1, hid_size) - 0.5
theta3 = np.random.rand(hid_size + 1, out_size) - 0.5

step = 0

while (step < 100):
    j = cost(theta1, theta2, theta3, X, y)
    t1grad, t2grad, t3grad = computeGrad(theta1, theta2, theta3, X, y)
    theta1 = theta1 - alpha * t1grad
    theta2 = theta2 - alpha * t2grad
    theta3 = theta3 - alpha * t3grad
    print("Step: " + str(step))
    print("Cost: " + str(j))
    step += 1


"""
for i in range(m):
    a1 = np.hstack((1, X[i]))
    z2 = a1 @ theta1
    a2 = np.hstack((1, sigmoid(z2)))
    z3 = a2 @ theta2
    a3 = np.hstack((1, sigmoid(z3)))
    z4 = a3 @ theta3
    a4 = sigmoid(z4)
    print(i)
    """