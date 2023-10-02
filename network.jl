# See LICENSE file for copyright and license details.

# network: weighted sums, activations, weights, biases, gradient
mutable struct NN
	dims::Vector{Int}

	act ::Function
	act′::Function

	   z::Vector{Vector{Float64}}
	   a::Vector{Vector{Float64}}
	   w::Vector{Matrix{Float64}}
	   b::Vector{Vector{Float64}}

	  ∇a::Vector{Vector{Float64}}
	  ∇w::Vector{Matrix{Float64}}
	  ∇b::Vector{Vector{Float64}}

	 Σ∇w::Vector{Matrix{Float64}}
	 Σ∇b::Vector{Vector{Float64}}
end

# batch of data points
Data = Vector{Tuple{Vector{Float64}, Vector{Float64}}}

# Rectified Linear Unit (ReLU)
relu(x)  = max(0, x)
relu′(x) = x < 0 ? 0 : 1

# leaky ReLU to avoid dead neurons
lrelu(x)  = max(0.01, x)
lrelu′(x) = x < 0 ? 0.01 : 1

# sigmoid
σ(x)  = 1 / (1 + exp(-x))
σ′(x) = σ(x) * (1 - σ(x))

# hyperbolic tangent
tanh(x)  = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
tanh′(x) = 1 - tanh(x)^2

# fill vectors and matrices
function init(dims::Vector{Int}; act=σ, act′=σ′)::NN
	len = length(dims)
	nn = NN(
		dims,
		act ,
		act′,
		Vector{Vector{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len),
		Vector{Matrix{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len),
		Vector{Matrix{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len),
		Vector{Matrix{Float64}}(undef, len),
		Vector{Vector{Float64}}(undef, len)
	)

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

function forward!(nn::NN)
	for i in 2:length(nn.dims)
		nn.z[i] = nn.w[i] * nn.a[i - 1] + nn.b[i]
		nn.a[i] = nn.act.(nn.z[i])
	end
	return nothing
end

loss(x, y)  = sum((x - y) .^ 2)
loss′(x, y) = 2 .* (x - y)

function backprop!(nn::NN, expected::Vector{Float64})
	len = length(nn.dims)
	nn.∇a[len] = loss′(nn.a[len], expected)

	for i in len:-1:2
		nn.∇b[i] = nn.act′.(nn.z[i]) .* nn.∇a[i]
		nn.∇w[i] = nn.a[i - 1]' .* nn.∇b[i]
		if i > 2
			nn.∇a[i - 1] = nn.w[i]' * nn.∇b[i]
		end
	end
	return nothing
end

# data: [(input, expected)], only one batch!
function train!(nn::NN, data::Data; η=1.0::Float64)::Float64
	fill!.(nn.Σ∇w, 0)
	fill!.(nn.Σ∇b, 0)

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
