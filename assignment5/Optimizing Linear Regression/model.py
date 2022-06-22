import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
from sklearn.datasets import make_regression
import numpy as np
import seaborn as sns


class Regressor:

    def __init__(self) -> None:
        self.X, self.y = self.generate_dataset(n_samples=200, n_features=1)
        n, d = self.X.shape
        self.w = np.zeros((d, 1))

    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """

        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=30)
        y = y.reshape(n_samples, 1)
        return X, y

    def linear_regression(self):
        """
        Performs linear regression on a dataset
        Returns:
            y: a numpy.ndarray of shape (n, 1) containing the predictions
        """
        y = np.dot(self.X, self.w)
        return y

    def predict(self, X):
        """
        Predicts the labels for a given dataset
        X: a numpy.ndarray of shape (n, d) containing the dataset
        Returns:
            y: a numpy.ndarray of shape (n,) containing the predictions
        """
        y = np.dot(X, self.w)
        return y

    def compute_loss(self):
        """
        Computes the MSE loss of a prediction
        Returns:
            loss: the loss of the prediction
        """
        predictions = self.linear_regression()
        tmp = np.square(np.subtract(predictions, self.y))
        loss = np.mean(tmp)
        return loss

    def compute_gradient(self):
        """
        Computes the gradient of the MSE loss
        Returns:
            grad: the gradient of the loss with respect to w
        """
        predictions = self.linear_regression()
        dif = (predictions - self.y)
        grad = 2 * np.dot(self.X.T, dif)
        return grad / self.X.shape[0]

    def fit(self, optimizer='gd', iteration_count=1000, render_animation=False, alpha=0.001, momentum=0.9, batch_size=5,
            g=0, epsilon=0.1, m=0, v=0, b1=0.9, b2=0.8):
        """
        Trains the model
        optimizer: the optimization algorithm to use
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        n_iters: the number of iterations to train for
        """

        figs = []
        loss = list()
        now_cost = 1e5
        counter = 0
        diff = 0
        self.b, self.m, self.g = 0.0, 0.0, g
        for i in range(1, iteration_count + 1):
            if optimizer == 'gd':
                self.gradient_descent(alpha)

            elif optimizer == "sgd":
                now_cost = self.sgd_optimizer(batch_size=batch_size, alpha=alpha)

            elif optimizer == "sgdMomentum" or optimizer == 'sgdm':
                now_cost = self.sgd_momentum(batch_size=batch_size, alpha=alpha)
                alpha *= momentum

            elif optimizer == "adagrad":
                self.adagrad_optimizer(self.g, epsilon)

            elif optimizer == "rmsprop":
                self.rmsprop_optimizer(self.g, alpha, epsilon)

            elif optimizer == "adam":
                self.adam_optimizer(m, v, b1, b2, epsilon)

            prev_cost = now_cost

            if optimizer[0] != 's':
                now_cost = self.compute_loss()

            if now_cost > prev_cost:
                counter += 1
            else:
                counter = 0

            if counter == 10:
                print('cost started to increase')
                break

            if abs(now_cost - prev_cost) < 0.001:
                diff += 1
            else:
                diff = 0

            if diff == 10:
                print("cost decreasing started to vanish")
                break

            loss.append(now_cost)
            if i % 10 == 0:
                print("Iteration: ", i)
                print("Loss: ", now_cost)

            if render_animation:
                fig = plt.figure()
                plt.scatter(self.X, self.y, color='red')
                plt.plot(self.X, self.predict(self.X), color='blue')
                plt.xlim(self.X.min(), self.X.max())
                plt.ylim(self.y.min(), self.y.max())
                plt.title(f'Optimizer:{optimizer}\nIteration: {i}')
                plt.close()
                figs.append(mplfig_to_npimage(fig))

        if render_animation and len(figs) > 0:
            clip = ImageSequenceClip(figs, fps=5)
            clip.write_gif(f'{optimizer}.gif', fps=5)

        self.plot_path(loss, optimizer)
        self.plot_scatter(optimizer)

    def gradient_descent(self, alpha):
        """
        Performs gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        self.w = self.w - alpha * self.compute_gradient()

    def sgd_optimizer(self, batch_size, alpha):
        """
        Performs stochastic gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """

        random_indexes = np.random.randint(0, len(self.X), batch_size)
        new_x = self.X[random_indexes, :]
        new_y = self.y[random_indexes]
        t = new_y - (self.m * new_x + self.b)
        self.b -= alpha * (-1 * 2 * t.sum() / len(new_x))
        self.m -= alpha * (-1 * 2 * new_x.T.dot(t).sum() / len(new_x))
        return np.sqrt(np.mean(np.square(np.subtract(self.y, self.m * self.X + self.b))))

    def sgd_momentum(self, alpha, batch_size):
        """
        Performs SGD with momentum to optimize the weights
        alpha: the learning rate
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        random_indexes = np.random.randint(0, len(self.X), batch_size)
        new_x = self.X[random_indexes, :]
        new_y = self.y[random_indexes]
        f = new_y - (self.m * new_x + self.b)
        self.m -= alpha * (-1 * 2 * new_x.T.dot(f).sum() / len(new_x))
        self.b -= alpha * (-1 * 2 * f.sum() / len(new_x))
        return np.sqrt(np.mean(np.square(np.subtract(self.y, self.m * self.X + self.b))))

    def adagrad_optimizer(self, g, epsilon):
        """
        Performs Adagrad optimization to optimize the weights
        alpha: the learning rate
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        grad = self.compute_gradient()
        self.g += grad ** 2
        self.w -= epsilon * grad / (1e-5 + np.sqrt(self.g))
        return self.w

    def rmsprop_optimizer(self, g, alpha, epsilon):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients
        alpha: the learning rate
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        grad = self.compute_gradient()
        self.w -= epsilon * grad / (np.sqrt(alpha * self.g + (1 - alpha) * grad ** 2) + 1e-5)
        return self.w

    def adam_optimizer(self, m, v, beta1, beta2, epsilon):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        alpha: the learning rate
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        grad = self.compute_gradient()
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        self.w -= epsilon * m / (np.sqrt(v) + 1e-5)
        return self.w

    def plot_scatter(self, optimizer):
        plt.scatter(self.X, self.y, color='red')
        plt.plot(self.X, self.predict(self.X), color='blue')
        plt.xlim(self.X.min(), self.X.max())
        plt.ylim(self.y.min(), self.y.max())
        plt.savefig(f'{optimizer}-scatter.png')
        plt.close()

    def plot_path(self, losses, label):
        """
        Plots the gradient descent path for the loss function
        Useful links: 
        -   http://www.adeveloperdiary.com/data-science/how-to-visualize-gradient-descent-using-contour-plot-in-python/
        -   https://www.youtube.com/watch?v=zvp8K4iX2Cs&list=LL&index=2
        """

        sns.lineplot(x=[i for i in range(len(losses))], y=losses, label=label)
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.savefig(f'{label}.png')
        plt.close()
        plt.show()
