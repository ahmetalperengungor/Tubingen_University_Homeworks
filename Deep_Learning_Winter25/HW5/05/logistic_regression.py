#!/usr/bin/python3
# -*- coding: utf-8 -*-
import csv

import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt


"""
logistic regression, assignment sheet 5

Authors: ADD YOUR NAMES HERE

complete the code sections marked with TODO
"""


def load_data(filepath : str) -> (np.ndarray,np.ndarray):
    """load the data from the given file, returning a matrix for X and a vector for y
    @param filepath: path to .csv file
    @return:  data, label tupel
    """
    xy = np.loadtxt(filepath, delimiter=',')
    x = xy[:, 0:2]
    y = xy[:, 2]
    return x, y


# (a)
def plot_data(inputs:np.ndarray, targets:np.ndarray, ax=None, colors=('blue', 'red')) -> plt.Axes:
    """ plots the data to a (possibly new) ax
    @param inputs: input data of shape TODO
    @param targets: labels of shape
    @param ax: plotting ax. If None a new one will created.
    @param colors: First element for label==0 second element for label==1.
    @return: plotting axis
    """
    if ax is None:
        # set up a new plot ax if you don't have one yet, otherwise you can plot to the existing one
        ax = plt.axes()
        # TODO set sensible x/y limits
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        # TODO set x/y labels and ticks
        plt.xlabel('Exam Score 1')
        plt.ylabel('Exam Score 2')
        plt.xticks(np.arange(0, 11, 1))
        plt.yticks(np.arange(0, 11, 1))
        plt.title('Student Admissions by Exam Scores')

    # TODO plot the data (e.g. via scatter)
    ax.scatter(inputs[targets==0, 1], inputs[targets==0, 2], c=colors[0], label='Class 0', alpha=0.5)
    ax.scatter(inputs[targets==1, 1], inputs[targets==1, 2], c=colors[1], label='Class 1', alpha=0.5)
    # TODO legend
    ax.legend()
    return ax


# (b)
def sigmoid(x:np.ndarray) -> np.ndarray:
    """
    @param x a scalar, vector or matrix
    @rtype: object
    """
    # TODO sigmoid function
    # TODO Do not use a for loop but solve it with the help of numpy broadcasting.
    # return np.zeros_like(x)
    return 1 / (1 + np.exp(-x))


# (c)
def cost(theta:np.ndarray, inputs:np.ndarray, targets:np.ndarray, epsilon:float=1e-10)->float:
    """ compute the cost function from the parameters theta
    @param theta: parameters to optimize of shape (10,)
    @param inputs: input data of shape (100,10)
    @param targets: labels of shape (100,)
    @param epsilon: factor for numerical stability
    @return: negative log likelihood
    """
    # TODO cost function
    # TODO add epsilon to the log expressions for numerical stability
    # TODO do not use a for loop but solve it with the help of numpy operators and broadcasting.
    # return 0.0
    predictions = sigmoid(inputs @ theta)
    cost_positive = -targets * np.log(predictions + epsilon)
    cost_negative = -(1 - targets) * np.log(1 - predictions + epsilon)
    total_cost = np.sum(cost_positive + cost_negative)
    return total_cost / len(targets)  # Average cost over all samples


# (c)
def gradient(theta, inputs, targets):
    """ compute the derivative of the cost function with respect to theta """
    # TODO compute the gradient with respect to the targets
    # TODO do not use a for loop but solve it with the help of numpy operators and broadcasting.
    # return np.zeros_like(theta)
    predictions = sigmoid(inputs @ theta)
    errors = predictions - targets
    grad = inputs.T @ errors
    return grad


# (d)
def gradient_descent(theta_0:np.ndarray, lr:float, steps:int, inputs:np.ndarray, targets:np.ndarray)->np.ndarray:
    """
    performs a gradient descent optimization for {steps} steps.
    Each 10000 step the current loss is printed.
    Args:
      theta_0: initial values of the parameters
      lr: learing rate
      steps: total number of iterations to perform
      inputs: training inputs
      targets: training targets
    returns the optimized values of theta
    """
    # TODO iteratively update theta for 'steps' iterations, here you are allowed to use one for loop.
    # return theta_0
    theta = theta_0.copy()
    for step in range(steps):
        grad = gradient(theta, inputs, targets)
        theta -= lr * grad
        
        if step % 10000 == 0:
            current_cost = cost(theta, inputs, targets)
            print(f"Step {step}, Cost: {current_cost:.4f}")
    
    return theta


# (e), (f)
def accuracy(inputs:np.ndarray, targets:np.ndarray, theta:np.ndarray)-> float:
    """ computes the accuracy (num elements correctly classified / total elements)
    @param theta: parameters to optimize of shape (10,)
    @param inputs: input data of shape (100,10)
    @param targets: labels of shape (100,)
    @return: the accuracy
    """
    # TODO calculate the accuracy
    # TODO do not use a for loop but solve it with the help of numpy operators and broadcasting.
    # Hint if the sigmoid is 0.5 then theta@X is 0
    # TODO calculate the accuracy
    # return 0.0
    predictions = sigmoid(inputs @ theta)
    predicted_labels = (predictions >= 0.5).astype(int)  # Convert probabilities to binary labels
    correct_predictions = np.sum(predicted_labels == targets)
    accuracy = correct_predictions / len(targets)
    return accuracy


# (e), (f)
def add_boundary(ax:plt.Axes, theta_trained:np.ndarray, polynomial_degree:int=1) -> None:
    """ adds the decision boundary to the axis
    @param ax: plotting axis on which the boundary is plotted
    @param theta_trained: optimized parameters of shape (10,)
    @param polynomial_degree: to apply polynomial extension else set it to 1
    @param targets: labels of shape (100,)
    """
    # TODO (e) plot a decision boundary on an existing plot ax (hint: use contour plot)
    # TODO (f) consider the polynomial extension
    # The following numpy methods might be useful: np.linspace, np.meshgrid, np.column_stack, ax.contour.
    # pass
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    grid_points = np.column_stack([np.ones(len(grid_points)), grid_points])  # Add bias term
    if polynomial_degree > 1:
        grid_points = polynomial_extension(grid_points, degree=polynomial_degree)
    Z = sigmoid(grid_points @ theta_trained)
    Z = Z.reshape(X.shape)
    ax.contour(X, Y, Z, levels=[0.5], colors='green', linestyles='dashed', linewidths=2)
    ax.set_title('Decision Boundary')


# <(f)
def polynomial_extension(inputs:np.ndarray, degree:int)->np.ndarray:
    """  . See exercise 1d
    @param inputs:
    @param degree: degree of the polynomial extension
    @return: A ndarray  where each column has the form x1^a*x2^b. A colum for each combination were a+b<=degree holds is created
    """
    # TODO calculate the polynomial extension
    # return inputs
    if degree < 1:
        raise ValueError("degree must be >= 1")

    n_samples, _ = inputs.shape
    # assume inputs already contain a bias column as first column
    x1 = inputs[:, 1]
    x2 = inputs[:, 2]

    features = []
    # iterate by total degree (0..degree). Within a given total degree, iterate
    # exponent of x1 from total_degree down to 0 to keep a stable ordering.
    for total_deg in range(0, degree + 1):
        for a in range(total_deg, -1, -1):
            b = total_deg - a
            if a == 0 and b == 0:
                term = np.ones(n_samples, dtype=float)
            else:
                term = (x1 ** a) * (x2 ** b)
            features.append(term)

    return np.column_stack(features)


def main():

    # TODO (f) test different values
    polynomial_degree = 2

    # load training and test sets
    train_inputs, train_targets = load_data('data_train.csv')
    test_inputs, test_targets = load_data("data_test.csv")

    # extend the input data in order to add a bias term to the dot product with theta
    train_inputs = np.column_stack([np.ones(len(train_targets)), train_inputs])
    test_inputs = np.column_stack([np.ones(len(test_targets)), test_inputs])

    print('-'*100, '\ninputs\n', train_inputs)
    print('-'*100, '\ntargets\n', train_targets)

    # (a) visualization
    ax = plot_data(train_inputs, train_targets, colors=('blue', 'red'), ax=None)

    train_inputs = polynomial_extension(train_inputs, degree=polynomial_degree)
    print('-'*100, '\ninputs (polynomial extension)\n', train_inputs, '\n', '-'*100)

    # (d) use these parameters for training the model
    theta_trained = gradient_descent(theta_0=np.zeros(len(train_inputs[0, :])),
                                     lr=1e-4,
                                     steps=100000,
                                     inputs=train_inputs,
                                     targets=train_targets)

    # (e) evaluation
    test_inputs = polynomial_extension(test_inputs, degree=polynomial_degree)
    ax = plot_data(test_inputs, test_targets, colors=('lightblue', 'orange'), ax=ax)
    print("Accuracy: " + str(accuracy(test_inputs, test_targets, theta_trained)))

    # (f) boundary plot
    add_boundary(ax=ax, theta_trained=theta_trained, polynomial_degree=polynomial_degree)
    plt.savefig("logistic_regression_result.png")
    plt.show()


if __name__ == '__main__':
    main()
