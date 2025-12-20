# Multilayer Perceptron

A simple multilayer perceptron (MLP), also known as a fully connected
feedforward artificial neural network, written from scratch in Julia.

## Usage

First, initialize the neural network by filling the `NN` structure from
[`mlp.jl`][1] according to the network's dimensions.

```julia
include("mlp.jl")

# size of each layer
dims = [784, 100, 20, 10]
nn = init(dims)
```

Then train the network on a data batch of type `Data`
(defined in [`mlp.jl`][1]).
The `train!()` function modifies the network's parameters based on the
average gradient across all data points.
Optionally, the learning rate `η` can be specified (default `η=1.0`).
The function returns the average loss of the network.

```julia
train!(nn, batch, η=0.001)
```

To achieve stochastic gradient descent, the `train!()` function can be
called from a `for`-loop.
The `forward!()` and `loss()` function can also be called manually.
See the [examples][2].

## Forward Pass and Gradient

Let $x \in \mathbb{R}^n$ be a dense layer's input vector,
$y \in \mathbb{R}^m$ the layer's output vector,
$W \in \mathbb{R}^{m \times n}$ the weight matrix,
$b \in \mathbb{R}^m$ the bias vector,
and $\sigma$ an element-wise activation function.
Define the pre-activation values as $z = b + Wx$, i.e.

```math
z_k = b_k + \sum_{j=1}^n w_{kj} \cdot x_j
```

for the forward pass $y_k = \sigma(z_k)$.
We can now calculate the gradient:

```math
\begin{aligned}
    \frac{\partial \mathcal{L}}{\partial z_k}
    &= \frac{\partial \mathcal{L}}{\partial y_k} \cdot \sigma'(z_k) \\
    \frac{\partial \mathcal{L}}{\partial b_k}
    &= \frac{\partial \mathcal{L}}{\partial z_k} \\
    \frac{\partial \mathcal{L}}{\partial w_{kj}}
    &= \frac{\partial \mathcal{L}}{\partial z_k} \cdot x_j \\
    \frac{\partial \mathcal{L}}{\partial x_j}
    &= \sum_{k=1}^m \frac{\partial \mathcal{L}}{\partial z_k} \cdot w_{kj}.
\end{aligned}
```

The `backprop!()` function from [`mlp.jl`][1] is optimized and
vectorized, so the equations look different than above.

[1]: ./mlp.jl
[2]: ./examples/
