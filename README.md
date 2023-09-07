# nn - neural network

This is a simple artificial neural network, specifically a multilayer
perceptron (MLP), written from scratch in Julia.

## Usage

First, initialize the neural network by filling the `NN` structure from
[network.jl][1] according to the networks dimensions.

```julia
include("network.jl")

# size of each layer
dims = [784, 100, 20, 10]
nn = init(dims)
```

Then train the network on a data batch of type `Data` (defined in
[network.jl][1]. The `train!()` function modifies the networks
parameters based on the average gradient across all data points.
Optionally, the learning rate `η` can be passed (default `η=1`). The
function returns the average loss of the network.

```julia
train!(nn, batch, η=0.001)
```

In order to achieve stochastic gradient descent, the `train!()` function
can be called from a `for`-loop. The `forward!()` and `loss()` function
can also be called manually. Have a look at the [examples][2].

## Gradient equations

![forward propagation equation](./forward.svg)

Based on the above equation, one can infer the partial derivatives of
the biases, weights and activations with respect to the loss / cost
using the chain rule.

![derivatives of biases, weights and activations](./gradient.svg)

The `backprop!()` function from [network.jl][1] is optimized and
vectorized, so the equations look different than above.

[1]: ./network.jl
[2]: ./examples/
