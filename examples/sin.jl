# See LICENSE file for copyright and license details.

import Printf

include("../network.jl")

dims = [1, 3, 3, 1]
len = length(dims)

n = init(dims)

# batched training data: [[(input, expected)]]
batches = [Data(undef, 1000) for i in 1:1000]
for batch in batches
	for j in eachindex(batch)
		x = rand() * pi * 1.5
		batch[j] = ([x], [sin(x)])
	end
end

for batch in batches
	Printf.@printf "Σcost = %.12f\n" train!(n, batch)
end

println("\nTesting with random values:\n---------------------------")
for i in 1:10
	n.a[1][1] = rand();
	forward!(n)
	Printf.@printf "sin(%.6f) = %.6f | NN: %.6f\n" n.a[1][1] sin(n.a[1][1]) n.a[len][1]
end
