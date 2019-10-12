import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
import random
import matplotlib.pyplot as plt
import time



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigGrad(z):
    a = sigmoid(z)
    return a * (1 - a)

def feedForward(theta1, theta2, x):
    a1 = np.hstack((1, x))
    z2 = a1 @ theta1
    a2 = np.hstack((1, sigmoid(z2)))
    z3 = a2 @ theta2
    a3 = sigmoid(z3)
    #print(a3)
    return np.argmax(a3)

def cost(theta1, theta2, X, y):
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1 @ theta1
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = a2 @ theta2
    a3 = sigmoid(z3)

    c = np.arange(10)
    yvec = (c == np.array([y]).T) * 1
    j = sum(sum(-yvec * np.log(a3) - (1-yvec) * np.log(1-a3))) / m

    return j

def computeGrad(theta1, theta2, X, y):
    m = X.shape[0]
    c = np.arange(10)
    yvec = (c == np.array([y]).T) * 1
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = a1 @ theta1
    a2 = np.hstack((np.ones((z2.shape[0], 1)), sigmoid(z2)))
    z3 = a2 @ theta2
    a3 = sigmoid(z3)

    j = sum(sum(-yvec * np.log(a3) - (1-yvec) * np.log(1-a3))) / m
    
    delta3 = a3 - yvec # (60000, 10)
    delta2 = delta3 @ theta2.T * np.hstack((np.ones((z2.shape[0], 1)), sigGrad(z2))) # (60000, 17)
    delta2 = delta2[:, 1:] # (60000, 16)

    # print(delta4.shape) #(60000, 10)
    # print(delta3.shape) #(60000, 16)
    # print(delta2.shape) #(60000, 16)
    # print(a3.shape) #(60000, 17)
    # print(a2.shape) #(60000, 17)
    # print(a1.shape) #(60000, 785)

    t2grad = (a2.T @ delta3) / m
    t1grad = (a1.T @ delta2) / m
    
    # print(theta3.shape) # (17, 10)
    # print(theta2.shape) # (17, 16)
    # print(theta1.shape) # (785, 16)

    # print(t3grad.shape) # (17, 10)
    # print(t2grad.shape) # (17, 16)
    # print(t1grad.shape) # (785, 16)

    return j, t1grad, t2grad

(xtrain, y), (xtest, ytest) = keras.datasets.mnist.load_data()

num_in = 784
hid_size = 16
out_size = 10
alpha = 0.0008

xtrain = xtrain / 256
xtest = xtest / 256
X = np.zeros((60000, 784))


for i in range(len(xtrain)):
    X[i] = xtrain[i].flatten()

m = X.shape[0]
epsilon = 0.2
mini_epsilon = 0.01

theta1 = (np.random.rand(num_in,   hid_size) - 0.5) * (epsilon * 2)
theta1bias = (np.random.rand(1, hid_size) - 0.5) * (mini_epsilon * 2)
theta2 = (np.random.rand(hid_size, out_size) - 0.5) * (epsilon * 2)
theta2bias = (np.random.rand(1, out_size) - 0.5) * (mini_epsilon * 2)

theta1 = np.vstack((theta1bias, theta1))
theta2 = np.vstack((theta2bias, theta2))

print(theta1.shape)
print(theta2.shape)


step = 0
sgd_size = 2000
avg = 0
counts = []
countsy = []

while (step < 100000000):

    if (step % 25 == 0):
        print(step)
    xind = random.randint(0, m-sgd_size-1)
    j, t1grad, t2grad = computeGrad(theta1, theta2, X[xind:xind+sgd_size], y[xind:xind+sgd_size])
    theta1 = theta1 - alpha * t1grad
    theta2 = theta2 - alpha * t2grad
    step += 1
    
    if (step % 100 == 0):
        index = random.randint(0, m-37)
        ps = [0] * 10
        count = 0
        for i in range(index, index + 36):
            prediction = feedForward(theta1, theta2, X[i])
            ans = y[i]
            if (prediction == ans):
                count += 1
            print("Prediction: " + str(prediction))
            print("Actual y:   " + str(ans))
            print("\n")
            ps[prediction] += 1
        print(ps)
        print(count/37)
        avg += count/37
        print(avg * 100/step)
        print(step)
        counts.append(count/37)
        countsy.append(step)
    
    if (step == 900000):
        plt.scatter(countsy, counts)
        plt.show()

    