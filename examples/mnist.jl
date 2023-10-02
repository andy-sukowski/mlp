# See LICENSE file for copyright and license details.

using Printf
using MLDatasets

include("../network.jl")

function one_hot(d::Int, n::Int)::Vector{Int}
	out = zeros(n)
	out[d] = 1
	return out
end

dims = [784, 32, 24, 16, 10]
nn = init(dims)

# load MNIST dataset from MLDatasets
train_x, train_y = MNIST(split=:train)[:]
inputs = vec.(copy.(eachslice(Float64.(train_x), dims=3)))
expected = Vector{Float64}.(one_hot.(train_y .+ 1, 10))
data = collect(zip(inputs, expected))::Data

# number of batches: 60000 / batch_size
batch_size = 10
batches = copy.(eachcol(reshape(data, batch_size, :)))

Σloss = Vector{Float64}(undef, length(batches))
@time for i in eachindex(batches)
	Σloss[i] = train!(nn, batches[i])
	@printf "Σloss[%d] = %.12f\n" i Σloss[i]
end

# load MNIST test dataset
test_x, test_y = MNIST(split=:test)[:]
test_inputs = vec.(copy.(eachslice(Float64.(test_x), dims=3)))

matches = 0
for i in eachindex(test_inputs)
	nn.a[1] = test_inputs[i]
	forward!(nn)
	global matches += argmax(last(nn.a)) - 1 == test_y[i]
end
@printf "\nAccuracy on test dataset: %.12f\n" matches / length(test_y)
