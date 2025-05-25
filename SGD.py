import numpy as np

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9):
        """
        Stochastic Gradient Descent optimizer with momentum.

        Args:
            params (list): List of parameters to optimize (numpy arrays).
            lr (float): Learning rate.
            momentum (float): Momentum factor.
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in self.params]

    def step(self, grads):
        """
        Perform a single optimization step.

        Args:
            grads (list): List of gradients for each parameter.
        """
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            self.params[i] += self.velocities[i]

    def zero_grad(self, grads):
        """
        Zero out gradients (if needed).

        Args:
            grads (list): List of gradients to zero.
        """
        for grad in grads:
            grad.fill(0)