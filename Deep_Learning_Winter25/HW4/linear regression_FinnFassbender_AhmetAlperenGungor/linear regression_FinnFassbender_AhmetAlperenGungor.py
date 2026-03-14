"""'
TODO fill in your names:
group member 1: Finn Fassbender
group member 2: Ahmet Alperen Güngör
"""

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

# Load the raw MNIST
X_train = datasets.MNIST("./", train=True, download=True).data.numpy()
y_train = datasets.MNIST("./", train=True, download=True).targets.numpy()

X_test = datasets.MNIST("./", train=False, download=True).data.numpy()
y_test = datasets.MNIST("./", train=False, download=True).targets.numpy()

# split eval data from train data:
eval_data_size = 10000
train_data_size = 50000
test_data_size = 10000

X_eval = X_train[0:10000, :, :]
y_eval = y_train[0:10000]
X_train = X_train[10000:, :, :]
y_train = y_train[10000:]
# As a sanity check, we print out the size of the training and test data.
print("Training data shape: ", X_train.shape)
print("Training labels shape: ", y_train.shape)
print("Evaluation data shape: ", X_eval.shape)
print("Evaluation labels shape: ", y_eval.shape)
print("Test data shape: ", X_test.shape)
print("Test labels shape: ", y_test.shape)

# Reshape the image data into rows
# IMPORTANT NOTE:
# In the lecture the so-called design matrix is defined to be the matrix
# with rows as the data points (in this exercise the flattened images).
# However, in the assignment sheet the design matrix is defined to be the
# matrix with columns as the data points

# Datatype float allows you to subtract images (is otherwise uint8)
X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype("float")
X_eval = np.reshape(X_eval, (X_eval.shape[0], -1)).astype("float")
X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype("float")
print("x shapes:")
print(X_train.shape, X_eval.shape, X_test.shape)
# normalize train data from range 0 to 255 to range 0 to 1
X_train = X_train / 255
X_eval = X_eval / 255
X_test = X_test / 255


# transform to y to one hot encoded vectors:
# each row is one y vector
def make_one_hot(v):
    """
    :param v: vector of the length of the dataset containing class labels from 0 to 9
    :return: a matrix of dim(lenght dataset,10), where the index of the corresponding label is set to one.
    """
    # TODO
    # one-hot using eye
    v_one_hot = np.eye(10)[v]
    return v_one_hot


y_train = make_one_hot(y_train)
y_eval = make_one_hot(y_eval)
y_test = make_one_hot(y_test)
print("y shapes:")
print(y_train.shape, y_eval.shape, y_test.shape)

# TODO for task e adapt the following parameters to achieve better results
batch_size = 100
epochs = 20
learning_rate = 0.1 # ORIGINAL 0.001

# usually one would use a random weight initialization, but for reproduceable results we use fixed weights
# Don't change these parameters
W = np.ones((784, 10)) * 0.01
b = np.ones((10)) * 0.01


def get_next_batch(iteraton, batch_size, data, label):
    X = data[iteraton * batch_size : (iteraton + 1) * batch_size, :]
    y = label[iteraton * batch_size : (iteraton + 1) * batch_size, :]
    return X, y


def softmax(x):
    """
    :param x The input dim(batch_size, 10)
    :return Result of the softmax dim(batch_size, 10)
    """
    # TODO calculate the softmax. Make sure to apply the stabilization techniques already discussed in the lecture and previous exercises.
    # Numerically stable softmax: subtract max per row
    # x shape: (batch_size, 10)
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    result = e_x / np.sum(e_x, axis=1, keepdims=True)
    return result


def cross_entropy_loss(y_hat, y):
    """
    :param y_hat is the output from fully connected layer dim(batch_size,10)
    :param y: is labels dim(batch_size,10)
    :return Loss dim(1)
    """
    # TODO calculate the cross entropy loss
    # clip predictions to avoid log(0)
    eps = 1e-12
    y_hat_clipped = np.clip(y_hat, eps, 1.0)
    # compute -sum(y * log(y_hat)) over classes and average
    loss = -np.sum(y * np.log(y_hat_clipped), axis=1)
    ce_loss = np.mean(loss)
    return ce_loss


def get_accuracy(y_hat, y):
    """
    the accuracy for one image is one if the maximum of y_hat has the same index as the 1 in y
    :param y_hat:  dim(batch_size,10)
    :param y: dim(batch_size,10)
    :return: mean accuracy dim(1)
    """
    acc = 0
    for row_y, row_y_hat in zip(y, y_hat):
        # TODO calc the accuracy:
        acc += (np.argmax(row_y) == np.argmax(row_y_hat)).astype(float)
    # return mean accuracy over given y
    return acc / batch_size


def do_network_inference(x):  # over whole batch
    """
    :param x: Input dim(batchsize,784)
    :return: Inference output dim(batchsize,10)
    """
    # TODO calculate y_hat without using a loop, note that the numpy has some special features that can be used.
    # z = x @ W + b. Use broadcasting for b
    z = x @ W + b
    y_hat = softmax(z)
    return y_hat


def get_delta_preactivations(y_hat, y):
    """
    :param y_hat: Inference result dim(batchsize,10)
    :param y: Ground truth dim(batchsize,10)
    :return: Delta preactivations dim(batchsize,10)
    """
    # TODO calculate delta_z
    # derivative of cross entropy loss (mean over batch) w.r.t. z is (y_hat - y) / N
    N = y.shape[0]
    delta_preactivations = (y_hat - y) / N
    return delta_preactivations


def get_delta_weights(y_hat, y, x_batch):
    """
    :param y_hat: Inference result dim(batchsize,10)
    :param y: Ground truth dim(batchsize,10)
    :param x_batch: Input data dim(batchsize,784)
    :return: Delta weights dim(784,10)
    """
    # TODO calculate delta_w without using a loop, note that numpy has some special features that can be used.
    # dW = X^T @ delta_z  (N normalisation is already inside delta_z)
    delta_z = get_delta_preactivations(y_hat, y)
    # x_batch shape (N,784) delta_z shape (N,10)
    delta_weights = x_batch.T @ delta_z
    return delta_weights


def get_delta_biases(y_hat, y):
    """
    :param y_hat: Inference result dim(batchsize,10)
    :param y: Ground truth dim(batchsize,10)
    :return: Delta biases dim(10)
    """
    # TODO calculate delta_b
    # db = sum(delta_z over batch)
    delta_z = get_delta_preactivations(y_hat, y)
    delta_biases = np.sum(delta_z, axis=0)
    return delta_biases


def do_parameter_update(delta_w, delta_b, W, b):
    """
    :param delta_w: dim(748,10)
    :param delta_b: dim(10)
    :param W: dim(748,10)
    :param b: dim(10)
    """
    # TODO update W and b
    # Gradient descent update: W -= lr * delta_w, b -= lr * delta_b
    W -= learning_rate * delta_w
    b -= learning_rate * delta_b


def numerical_grad(func, x, eps=1e-6):
    """
    Generic numerical gradient of scalar-valued func w.r.t. array x using forward finite differences.
    :param func: callable taking an array like x and returning a scalar
    :param x: numpy array to differentiate w.r.t.
    :param eps: small perturbation
    :return: numpy array same shape as x containing numerical gradients
    """
    base = func(x)
    grad = np.zeros_like(x, dtype=float)

    # iterate over all indices and compute forward finite differences
    for idx in np.ndindex(x.shape):
        # TODO implement the numerical gradient calculation
        # forward finite differences
        old = x[idx]
        x[idx] = old + eps
        f_plus = func(x)
        # restore
        x[idx] = old
        grad[idx] = (f_plus - base) / eps

    return grad


################################################################################
###  Comparing Analytical and Numerical Gradients
################################################################################

# Small gradient check using the first training batch (uses global batch_size)
x_chk, y_chk = get_next_batch(0, batch_size, X_train, y_train)
z_chk = x_chk @ W + b
y_hat_chk = softmax(z_chk)

# analytical vs. numerical gradient wrt. preactivations z:
print("\n### Gradient check w.r.t. preactivations (z):")
analytical_z = get_delta_preactivations(y_hat_chk, y_chk)
numerical_z = numerical_grad(
    lambda z: cross_entropy_loss(softmax(z), y_chk), z_chk
)
abs_diff_z = np.abs(analytical_z - numerical_z)
print(f"max abs diff: {np.max(abs_diff_z):.6e}, mean abs diff: {np.mean(abs_diff_z):.6e}") # fmt: skip
print("example analytical grad [0,:4]:", analytical_z[0, :4])
print("example numerical  grad [0,:4]:", numerical_z[0, :4])
print("example abs diff        [0,:4]:", abs_diff_z[0, :4])


# analytical vs. numerical gradient wrt. W:
print("\n### Gradient check w.r.t. weights (W):")
analytical_W = get_delta_weights(y_hat_chk, y_chk, x_chk)
numerical_W = numerical_grad(
    lambda W_: cross_entropy_loss(softmax(x_chk @ W_ + b), y_chk), W
)
abs_diff_W = np.abs(analytical_W - numerical_W)
print(f"max abs diff: {np.max(abs_diff_W):.6e}, mean abs diff: {np.mean(abs_diff_W):.6e}") # fmt: skip
print("example analytical grad [200:204, 0:4]:\n", analytical_W[200:204, 0:4])
print("example numerical  grad [200:204, 0:4]:\n", numerical_W[200:204, 0:4])
print("example abs diff         [200:204, 0:4]:\n", abs_diff_W[200:204, 0:4])


# analytical vs. numerical gradient wrt. b:
print("\n### Gradient check w.r.t. biases (b):")
analytical_b = get_delta_biases(y_hat_chk, y_chk)
numerical_b = numerical_grad(
    lambda b_: cross_entropy_loss(softmax(x_chk @ W + b_), y_chk), b
)
abs_diff_b = np.abs(analytical_b - numerical_b)
print(f"max abs diff: {np.max(abs_diff_b):.6e}, mean abs diff: {np.mean(abs_diff_b):.6e}") # fmt: skip
print("example analytical grad [0,:4]:", analytical_b[:4])
print("example numerical  grad [0,:4]:", numerical_b[:4])
print("example abs diff        [0,:4]:", abs_diff_b[:4])
print("\n")


################################################################################
###  Model Training
################################################################################

# do training and evaluation
mean_eval_losses = []
mean_train_losses = []
mean_eval_accs = []
mean_train_accs = []

for epoch in range(epochs):
    # training
    mean_train_loss_per_epoch = 0
    mean_train_acc_per_epoch = 0
    for i in range(train_data_size // batch_size):
        x, y = get_next_batch(i, batch_size, X_train, y_train)
        y_hat = do_network_inference(x)
        train_loss = cross_entropy_loss(y_hat, y)
        train_accuracy = get_accuracy(y_hat, y)
        delta_w = get_delta_weights(y_hat, y, x)
        delta_b = get_delta_biases(y_hat, y)

        do_parameter_update(delta_w, delta_b, W, b)
        mean_train_loss_per_epoch += train_loss
        mean_train_acc_per_epoch += train_accuracy
        # print("epoch: {0:d} \t iteration {1:d} \t train loss: {2:f}".format(epoch, i,train_loss))

    mean_train_loss_per_epoch = mean_train_loss_per_epoch / (
        (train_data_size // batch_size)
    )
    mean_train_acc_per_epoch = mean_train_acc_per_epoch / (
        (train_data_size // batch_size)
    )
    print(
        f"epoch:{epoch:d} \t "
        f"mean train loss: {mean_train_loss_per_epoch:f} \t "
        f"mean train acc: {mean_train_acc_per_epoch:f}"
    )

    # evaluation:
    mean_eval_loss_per_epoch = 0
    mean_eval_acc_per_epoch = 0
    # TODO calculate the evaluation loss and accuracy (similar to the training loop)
    for i in range(eval_data_size // batch_size):
        x, y = get_next_batch(i, batch_size, X_eval, y_eval)
        y_hat = do_network_inference(x)
        mean_eval_loss_per_epoch += cross_entropy_loss(y_hat, y)
        mean_eval_acc_per_epoch += get_accuracy(y_hat, y)

    mean_eval_loss_per_epoch = mean_eval_loss_per_epoch / (
        eval_data_size // batch_size
    )
    mean_eval_acc_per_epoch = mean_eval_acc_per_epoch / (
        (eval_data_size // batch_size)
    )
    print(
        f"epoch:{epoch:d} \t "
        f"mean eval loss: {mean_eval_loss_per_epoch:f} \t "
        f"mean eval acc: {mean_eval_acc_per_epoch:f}"
    )
    mean_eval_losses.append(mean_eval_loss_per_epoch)
    mean_train_losses.append(mean_train_loss_per_epoch)
    mean_eval_accs.append(mean_eval_acc_per_epoch)
    mean_train_accs.append(mean_train_acc_per_epoch)

# testing
mean_test_loss_per_epoch = 0
mean_test_acc_per_epoch = 0
# TODO calculate the test loss and accuracy
for i in range(test_data_size // batch_size):
    x, y = get_next_batch(i, batch_size, X_test, y_test)
    y_hat = do_network_inference(x)
    mean_test_loss_per_epoch += cross_entropy_loss(y_hat, y)
    mean_test_acc_per_epoch += get_accuracy(y_hat, y)


mean_test_loss_per_epoch = mean_test_loss_per_epoch / (
    test_data_size // batch_size
)
mean_test_acc_per_epoch = mean_test_acc_per_epoch / (
    (test_data_size // batch_size)
)
print(
    f"final test loss: {mean_test_loss_per_epoch:.3f} \t "
    f"final test acc: {mean_test_acc_per_epoch:.3f}"
)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(epochs), mean_train_losses, "r", label="train loss")
ax1.plot(range(epochs), mean_eval_losses, "b", label="eval loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2.plot(range(epochs), mean_train_accs, "r", label="train acc")
ax2.plot(range(epochs), mean_eval_accs, "b", label="eval acc")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()
# plt.show()
plt.savefig("assignment04_linear_regression.png")
