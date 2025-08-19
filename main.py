import math
import random

def sigmoid(z):
    return 1 / ( 1 + math.exp(-z) )

def forward_pass(w1, x1, w2, x2, b):
    z = w1 * x1 + w2 * x2 + b
    a = sigmoid(z)
    return a

def compute_loss(y_true, y_pred):
    epsilon = 10**(-12)
    a = min(max(y_pred, epsilon), 1 - epsilon) 
    loss = -(y_true * math.log(a) + (1 - y_true) * math.log(1 - a))
    return loss

def update_weights(w1, w2, b, x1, x2, y_true, y_pred, learning_rate):
    error = y_pred - y_true
    dw1 = error * x1
    dw2 = error * x2
    db = error

    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b = b - learning_rate * db

    return w1, w2, b

def train(X, Y, epochs, learning_rate):
    w1 = random.uniform(-0.5, 0.5)
    w2 = random.uniform(-0.5, 0.5)
    b = random.uniform(-0.5, 0.5)

    for epoch in range(epochs):
        print("__________")
        print("epoch " , epoch)
        loss_y_true = []
        loss_y_pred = []

        weights1 = []
        weights2 = []
        bias = []

        for (x1, x2), y_true in zip(X, Y):
            y_pred = forward_pass(w1, x1, w2, x2, b)
            w1, w2, b = update_weights(w1, w2, b, x1, x2, y_true, y_pred, learning_rate)
            
            weights1.append(w1)
            weights2.append(w2)
            bias.append(b)
            loss_y_true.append(y_true)
            loss_y_pred.append(y_pred)

        print("weight 1: ", sum(weights1) / len(weights1), 
              "weight 2:", sum(weights2) / len(weights2), 
              "bias: ", sum(bias) / len(bias))
        print("loss: ", 
              compute_loss(sum(loss_y_true) / len(loss_y_true), 
                           sum(loss_y_pred) / len(loss_y_pred)))
    return w1, w2, b

def predict(X, Y, w1, w2, b):
    index = 0
    for (x1, x2) in X:
        prediction = forward_pass(w1, x1, w2, x2, b)
        print("prediction: ", round(prediction, 2), "| actual: ", Y[index])
        index += 1

def AND_gate():
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Y = [0, 0, 0, 1]

    (w1, w2, b) = train(X, Y, 500, 0.5)
    
    predict(X, Y, w1, w2, b)

def OR_gate():
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Y = [0, 1, 1, 1]

    (w1, w2, b) = train(X, Y, 500, 0.5)

    predict(X, Y, w1, w2, b)

def NAND_gate():
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Y = [1, 1, 1, 0]

    (w1, w2, b) = train(X, Y, 500, 0.5)
    
    predict(X, Y, w1, w2, b)

def NOR_gate():
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Y = [1, 0, 0, 0]

    (w1, w2, b) = train(X, Y, 500, 0.5)

    predict(X, Y, w1, w2, b)

def predict_with_hidden_layer(X, Y, Y_AND, Y_OR,  epochs, loss):
    (w11, w12, b1) = train(X, Y_AND, epochs, loss) # AND
    (w21, w22, b2) = train(X, Y_OR, epochs, loss) # OR

    H = []

    # Hidden layer, combined AND and OR for XOR
    for (x1, x2) in X:
        prediction_AND = forward_pass(w11, x1, w12, x2, b1)
        prediction_OR = forward_pass(w21, x1, w22, x2, b2)
        H.append((prediction_AND, prediction_OR))

    # Train new weights with result from hidden layer 
    (w1, w2, b) = train(H, Y, epochs, loss) # XOR

    index = 0
    for (h1, h2) in H:
        prediction = forward_pass(w1, h1, w2, h2, b)
        print("prediction: ", round(prediction, 2), "| actual: ", Y[index])
        index += 1

def XOR_gate():
    X = [(0, 0), (0, 1), (1, 0), (1, 1)]
    Y = [0, 1, 1, 0]

    Y_AND = [0, 0, 0, 1]
    Y_OR = [0, 1, 1, 1]

    predict_with_hidden_layer(X, Y, Y_AND, Y_OR, epochs=500, loss=0.5)

XOR_gate()