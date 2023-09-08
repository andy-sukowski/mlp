# See LICENSE file for copyright and license details.

using Printf

include("../network.jl")

dims = [2, 5, 1]
nn = init(dims)

# batched training data: [[(input, expected)]]
batches = [Data(undef, 10) for i in 1:100000]
for batch in batches
	for j in eachindex(batch)
		in = rand(2) ./ 2 # because σ: ℝ → (0, 1)
		batch[j] = (in, [sum(in)])
	end
end

Σloss = Vector{Float64}(undef, length(batches))
for i in eachindex(batches)
	Σloss[i] = train!(nn, batches[i], η=5)
	@printf "Σloss[%d] = %.12f\n" i Σloss[i]
end

println("\nTesting with random values:\n---------------------------")
for i in 1:10
	nn.a[1] = rand(2) ./ 2
	forward!(nn)
	@printf "%.6f + %.6f = %.6f | NN: %.6f\n" nn.a[1][1] nn.a[1][2] sum(nn.a[1]) last(nn.a)[1]
end
