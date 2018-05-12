import pandas as pd
import numpy as np

def initialize_parameters_noisy_circles(n_x, n_h, n_y):
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,
        "b1": np.zeros((n_h, 1)),
        "W2": np.random.rand(n_y, n_h) * 0.01,
        "b2": np.zeros((n_y, 1))
    }


# return activation
def forward_propagation_noisy_circles(X, feature_weights):
    W1 = feature_weights["W1"]
    b1 = feature_weights["b1"]
    W2 = feature_weights["W2"]
    b2 = feature_weights["b2"]

    # W1:5*5, X:5*375 +
    Z1 = np.dot(W1, X) + b1  # W1:n_h*n_x , X:n_x*m, therefore Z1 will be n_h*m
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    activations = {
        "Z1": Z1,  # W1:n_h*n_x , X:n_x*m, therefore Z1 will be n_h*m
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, activations


def compute_cost_noisy_circles(A2, Y, feature_weights):
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y)

    logprobs = logprobs + np.multiply(np.log(1 - A2), 1 - Y)

    cost = -1 * np.sum(logprobs.T) / m
    print(cost)
    cost = cost.squeeze()

    assert (isinstance(cost, float))

    return cost


def backward_propagation_noisy_circles(feature_weights, activations, X, Y):
    m = X.shape[1]
    W1 = feature_weights["W1"]
    W2 = feature_weights["W2"]

    A1 = activations['A1']
    A2 = activations['A2']

    dZ2 = A2 - Y  # size dz2: 1*m
    dW2 = np.dot(dZ2, A1.T) / m  # dw2: 1*4

    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # size 1*1
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))  # size 4*m,
    dW1 = np.dot(dZ1, X.T) / m  # size:4*n
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # size 4*1

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters_noisy_circles(feature_weights, grads, learning_rate=.00002):
    W1 = feature_weights['W1']
    b1 = feature_weights['b1']
    W2 = feature_weights['W2']
    b2 = feature_weights['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    update_feature_weights = {"W1": W1 - learning_rate * dW1,
                              "b1": b1 - learning_rate * db1,
                              "W2": W2 - learning_rate * dW2,
                              "b2": b2 - learning_rate * db2}

    return update_feature_weights


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def nn_model_noisy_circles(X, Y, n_h, num_iterations=10, print_cost=False, hidden_units=5):
    np.random.seed(3)
    feature_weights = initialize_parameters_noisy_circles(X.shape[0], hidden_units, Y.shape[0])
    W1 = feature_weights['W1']
    b1 = feature_weights['b1']
    W2 = feature_weights['W2']
    b2 = feature_weights['b2']

    for i in range(0, num_iterations):
        A2, activations = forward_propagation_noisy_circles(X, feature_weights)
        cost = compute_cost_noisy_circles(A2, Y, feature_weights)
        grads = backward_propagation_noisy_circles(feature_weights, activations, X, Y)

        feature_weights = update_parameters_noisy_circles(feature_weights, grads, 1.8)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return feature_weights


def predict_noisy_circles(parameters, X):
    A2, activations = forward_propagation_noisy_circles(X, parameters)  # A2 size will be n_y*n
    predictions = A2 > 0.5

    return predictions