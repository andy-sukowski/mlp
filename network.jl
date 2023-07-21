# See LICENSE file for copyright and license details.

# network: weighted sums, activations, weights, biases, gradient
mutable struct NN
	dims :: Vector{Int}

	   z :: Vector{Vector{Float64}}
	   a :: Vector{Vector{Float64}}
	   w :: Vector{Matrix{Float64}}
	   b :: Vector{Vector{Float64}}

	  ∇a :: Vector{Vector{Float64}}
	  ∇w :: Vector{Matrix{Float64}}
	  ∇b :: Vector{Vector{Float64}}

	 Σ∇w :: Vector{Matrix{Float64}}
	 Σ∇b :: Vector{Vector{Float64}}
end

Data = Vector{Tuple{Vector{Float64}, Vector{Float64}}}

# fill vectors and matrices
function init(dims :: Vector{Int}) :: NN
	len = length(dims)
	nn = NN(dims,
		Vector{Vector{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len),
		Vector{Matrix{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len),
		Vector{Matrix{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len),
		Vector{Matrix{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len))

	nn.a[1] = Vector{Float64}(undef, dims[1])
	# only nn.a has first element, other vectors are shifted by 1
	nn.z[1] = nn.a[1] = nn.∇a[1] = nn.b[1] = nn.∇b[1] = nn.Σ∇b[1] = []
	nn.w[1] = nn.∇w[1] = nn.Σ∇w[1] = [;;]

	for i in 2:length(dims)
		nn.z[i]   = Vector{Float64}(undef, dims[i])
		nn.a[i]   = Vector{Float64}(undef, dims[1])
		nn.∇a[i]  = Vector{Float64}(undef, dims[i])

		nn.w[i]   = randn(dims[i], dims[i - 1])
		nn.∇w[i]  = Matrix{Float64}(undef, dims[i], dims[i - 1])
		nn.Σ∇w[i] = Matrix{Float64}(undef, dims[i], dims[i - 1])

		nn.b[i]   = zeros(dims[i])
		nn.∇b[i]  = Vector{Float64}(undef, dims[i])
		nn.Σ∇b[i] = Vector{Float64}(undef, dims[i])
	end

	return nn
end

# leaky ReLU to avoid dead neurons
ReLU(x)  = max(0.01 * x, x)
ReLU′(x) = x >= 0 ? 1 : 0.01

σ(x)  = 1 / (1 + exp(-x))
σ′(x) = σ(x) * (1 - σ(x))

const act  = σ
const act′ = σ′

function forward!(nn :: NN)
	for i in 2:length(nn.dims)
		nn.z[i] = nn.w[i] * nn.a[i - 1] + nn.b[i]
		nn.a[i] = act.(nn.z[i])
	end
	return nothing
end

loss(x, y)  = sum((x - y) .^ 2)
loss′(x, y) = 2 .* (x - y)

function backprop!(nn :: NN, expected :: Vector{Float64})
	len = length(nn.dims)

	nn.∇a[len] = loss′(nn.a[len], expected)

	for i in len:-1:2
		nn.∇b[i] = act′.(nn.z[i]) .* nn.∇a[i]
		nn.∇w[i] = transpose(nn.a[i - 1]) .* nn.∇b[i]
		if i > 2
			nn.∇a[i - 1] = transpose(nn.w[i]) * nn.∇b[i]
		end
	end
	return nothing
end

# data: [(input, expected)], only one batch!
function train!(nn :: NN, data :: Data; η = 1 :: Float64) :: Float64
	for i in 2:length(nn.dims)
		nn.Σ∇w[i] .= 0
		nn.Σ∇b[i] .= 0
	end

	Σloss = 0

	for d in data
		nn.a[1] = d[1]
		forward!(nn)
		Σloss += loss(nn.a[length(nn.dims)], d[2]) / length(data)

		backprop!(nn, d[2])
		nn.Σ∇w += nn.∇w / length(data)
		nn.Σ∇b += nn.∇b / length(data)
	end

	# play around with learning rate η
	nn.w -= η * nn.Σ∇w
	nn.b -= η * nn.Σ∇b

	return Σloss
end
