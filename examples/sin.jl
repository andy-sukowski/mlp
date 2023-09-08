# See LICENSE file for copyright and license details.

using Printf

include("../network.jl")

dims = [1, 10, 10, 10, 1]
nn = init(dims)

# batched training data: [[(input, expected)]]
batches = [Data(undef, 5) for i in 1:100000]
for batch in batches
	for j in eachindex(batch)
		x = rand() * pi * 2
		batch[j] = ([x], [sin(x) / 2 + 0.5])
	end
end

Σloss = Vector{Float64}(undef, length(batches))
for i in eachindex(batches)
	Σloss[i] = train!(nn, batches[i], η=5)
	@printf "Σloss[%d] = %.12f\n" i Σloss[i]
end

println("\nTesting with random values:\n---------------------------")
for i in 1:10
	nn.a[1][1] = rand() * pi * 2
	forward!(nn)
	expected = sin(nn.a[1][1]) / 2 + 0.5
	@printf "sin(%.6f) = %.6f | NN: %.6f\n" nn.a[1][1] expected last(nn.a)[1]
end
