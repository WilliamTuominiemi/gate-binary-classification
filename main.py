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
        print("epoch " , epoch)
        loss_y_true = []
        loss_y_pred = []

        for (x1, x2), y_true in zip(X, Y):
            y_pred = forward_pass(w1, x1, w2, x2, b)
            w1, w2, b = update_weights(w1, w2, b, x1, x2, y_true, y_pred, learning_rate)
            loss_y_true.append(y_true)
            loss_y_pred.append(y_pred)

        print("loss: ", 
              compute_loss(sum(loss_y_true) / len(loss_y_true), 
                           sum(loss_y_pred) / len(loss_y_pred)))
    return w1, w2, b

X = [(0, 0), (0, 1), (1, 0), (1, 1)]

Y = [0, 0, 0, 1] # AND
# Y = [0, 1, 1, 1] # OR

(w1, w2, b) = train(X, Y, 200, 0.2)

index = 0
for (x1, x2) in X:
    prediction = forward_pass(w1, x1, w2, x2, b)
    print(prediction, Y[index])
    index += 1