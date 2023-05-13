# See LICENSE file for copyright and license details.

using Printf

include("../network.jl")

dims = [1, 10, 10, 10, 1]
len = length(dims)

n = init(dims)

# batched training data: [[(input, expected)]]
batches = [Data(undef, 5) for i in 1:100000]
for batch in batches
	for j in eachindex(batch)
		x = rand() * pi * 2
		batch[j] = ([x], [sin(x) / 2 + 0.5])
	end
end

for batch in batches
	@printf "Σcost = %.12f\n" train!(n, batch)
end

println("\nTesting with random values:\n---------------------------")
for i in 1:10
	n.a[1][1] = rand() * pi * 2
	forward!(n)
	expected = sin(n.a[1][1]) / 2 + 0.5
	@printf "sin(%.6f) = %.6f | NN: %.6f\n" n.a[1][1] expected n.a[len][1]
end
