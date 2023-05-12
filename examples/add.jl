# See LICENSE file for copyright and license details.

import Printf

include("../network.jl")

dims = [2, 5, 1]
len = length(dims)

# weighted sums, activations, weights, biases
z = [];  a = [];  w = [];  b = []
        ∇a = []; ∇w = []; ∇b = [] # gradient

# average gradient for batch
Σ∇w = []; Σ∇b = []

init(dims, z, a, w, b, ∇a, ∇w, ∇b, Σ∇w, Σ∇b)

# batched training data: [[(input, expected)]]
batches = [[] for i = 1:1000]
for batch in batches
	for j in 1:100
		in = rand(2) ./ 2 # because σ: R → [0; 1]
		push!(batch, (in, [sum(in)]))
	end
end

for batch in batches
	Printf.@printf "Σcost = %.12f\n" train(dims, batch, z, a, w, b, ∇a, ∇w, ∇b, Σ∇w, Σ∇b)
end

println("\nTesting with random values:\n---------------------------")
for i in 1:10
	a[1] = rand(2) ./ 2
	forward(len, z, a, w, b)
	Printf.@printf "%.6f + %.6f = %.6f | NN: %.6f\n" a[1][1] a[1][2] a[1][1]+a[1][2] a[len][1]
end
