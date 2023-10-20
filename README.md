# Multilayer Perceptron

A simple multilayer perceptron (MLP), also known as a fully connected
feedforward artificial neural network, written from scratch in Julia.

## Usage

First, initialize the neural network by filling the `NN` structure from
[mlp.jl][1] according to the networks dimensions.

```julia
include("mlp.jl")

# size of each layer
dims = [784, 100, 20, 10]
nn = init(dims)
```

Then train the network on a data batch of type `Data` (defined in
[mlp.jl][1]). The `train!()` function modifies the networks parameters
based on the average gradient across all data points. Optionally, the
learning rate `η` can be passed (default `η=1.0`). The function returns
the average loss of the network.

```julia
train!(nn, batch, η=0.001)
```

In order to achieve stochastic gradient descent, the `train!()` function
can be called from a `for`-loop. The `forward!()` and `loss()` function
can also be called manually. Have a look at the [examples][2].

## Gradient equations

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./images/forward.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./images/forward_inv.svg">
  <img alt="forward propagation equation" src="./images/forward.svg">
</picture>

Based on the above equation, one can infer the partial derivatives of
the biases, weights and activations with respect to the loss / cost
using the chain rule.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./images/gradient.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./images/gradient_inv.svg">
  <img alt="derivatives of biases, kernels and activations" src="./images/gradient.svg">
</picture>

The `backprop!()` function from [mlp.jl][1] is optimized and
vectorized, so the equations look different than above.

[1]: ./mlp.jl
[2]: ./examples/
