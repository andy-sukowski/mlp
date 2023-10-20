# See LICENSE file for copyright and license details.

using Printf
using ProgressMeter

include("../mlp.jl")

dims = [1, 10, 10, 10, 1]
nn = init(dims, act=tanh, act′=tanh′)

# batched training data: [[(input, expected)]]
batches = [Data(undef, 5) for i in 1:100000]
for batch in batches
	for j in eachindex(batch)
		x = rand() * 2 * π - π
		batch[j] = ([x], [sin(x)])
	end
end

# average loss for each batch
Σlosses = Vector{Float64}(undef, length(batches))

p = Progress(length(batches); desc="Training:", dt=0.1, barlen=16)
for i in eachindex(batches)
	Σlosses[i] = train!(nn, batches[i], η=0.15)
	next!(p; showvalues = [(:batch, i),
		(:loss, @sprintf("%0.16f", Σlosses[i]))])
end
finish!(p)
