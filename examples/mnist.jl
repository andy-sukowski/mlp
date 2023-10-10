# See LICENSE file for copyright and license details.

using Printf
using MLDatasets: MNIST
using ProgressMeter

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

# average loss for each batch
Σlosses = Vector{Float64}(undef, length(batches))

p = Progress(length(batches); desc="Training:", dt=0.1, barlen=16)
for i in eachindex(batches)
	Σlosses[i] = train!(nn, batches[i])
	next!(p; showvalues = [(:batch, i),
		(:loss, @sprintf("%0.16f", Σlosses[i]))])
end
finish!(p)

# load MNIST test dataset
test_x, test_y = MNIST(split=:test)[:]
test_inputs = vec.(copy.(eachslice(Float64.(test_x), dims=3)))

matches = 0
for i in eachindex(test_inputs)
	nn.a[1] = test_inputs[i]
	forward!(nn)
	global matches += argmax(last(nn.a)) - 1 == test_y[i]
end
println("Accuracy on test dataset: ", matches / length(test_y))
