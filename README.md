# Neural Network Logic Gates

This project implements a simple neural network from scratch to model basic logic gates (AND, OR, NAND, NOR) and demonstrates how to create an XOR gate using a hidden layer.

## Overview

The code implements a single-layer perceptron for basic logic operations and a two-layer network for the XOR problem, which is not linearly separable.

## Key Components

### Core Functions

- sigmoid(): Activation function that maps any value to a range between 0 and 1

- forward_pass(): Performs the forward propagation through the network

- compute_loss(): Calculates the binary cross-entropy loss

- update_weights(): Updates weights using gradient descent

- train(): Main training loop that iterates through epochs

### Logic Gate Implementations

- AND_gate(): Implements the AND logic operation

- OR_gate(): Implements the OR logic operation

- NAND_gate(): Implements the NAND logic operation

- NOR_gate(): Implements the NOR logic operation

- XOR_gate(): Implements XOR using a hidden layer with AND and OR gates

## How It Works

For basic gates (AND, OR, NAND, NOR): The network learns appropriate weights through training

For XOR: Uses a hidden layer that combines AND and OR operations, then trains a final layer on these features
