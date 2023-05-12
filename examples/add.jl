# See LICENSE file for copyright and license details.

import Printf

include("../network.jl")

dims = [2, 5, 1]
len = length(dims)

n = init(dims)

# batched training data: [[(input, expected)]]
batches = [Data(undef, 100) for i in 1:1000]
for batch in batches
	for j in eachindex(batch)
		in = rand(2) ./ 2 # because σ: R → [0; 1]
		batch[j] = (in, [sum(in)])
	end
end

for batch in batches
	Printf.@printf "Σcost = %.12f\n" train!(n, batch)
end

println("\nTesting with random values:\n---------------------------")
for i in 1:10
	n.a[1] = rand(2) ./ 2
	forward!(n)
	Printf.@printf "%.6f + %.6f = %.6f | NN: %.6f\n" n.a[1][1] n.a[1][2] n.a[1][1]+n.a[1][2] n.a[len][1]
end
