import math
import random

def sigmoid(z):
    return 1 / ( 1 + (math.e)^(-z) )

def forward_pass(w1, x1, w2, x2, b):
    z = w1 * x1 + w2 * x2 + b
    a = sigmoid(z)
    return a

def compute_loss(y_true, y_pred):
    loss = -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))
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
    w1 = random.randint(0, 10)
    w2 = random.randint(0, 10)
    b = random.randint(0, 10)

    for epoch in range(epochs):
        for x1, x2, y_true in zip(X, Y):
            y_pred = forward_pass(w1, x1, w2, x2, b)
            w1, w2, b = update_weights(w1, w2, b, x1, x2, y_true, y_pred, learning_rate)
    return w1, w2, b

