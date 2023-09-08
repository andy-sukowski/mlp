# See LICENSE file for copyright and license details.

using Printf
using MLDatasets

include("../network.jl")

function one_hot(d :: Int, n :: Int) :: Vector{Int}
	out = zeros(n)
	out[d] = 1
	return out
end

dims = [784, 32, 24, 16, 10]
nn = init(dims)

# number of batches: 60000 / batch_size
batch_size = 10

# Loading MNIST dataset from MLDatasets
train_x, train_y = MNIST(split=:train)[:]
inputs = vec.(copy.(eachslice(Float64.(train_x), dims=3)))
expected = Vector{Float64}.(one_hot.(train_y .+ 1, 10))
data = collect(zip(inputs, expected)) :: Data
batches = copy.(eachcol(reshape(data, batch_size, :)))

Σloss = Vector{Float64}(undef, length(batches))
for i in eachindex(batches)
	Σloss[i] = train!(nn, batches[i])
	@printf "Σloss[%d] = %.12f\n" i Σloss[i]
end
