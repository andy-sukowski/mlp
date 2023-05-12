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

# fill vectors and matrices, needs improvement
function init(dims :: Vector{Int}) :: NN
	n = NN([], [], [], [], [], [], [], [], [], [])
	n.dims = dims

	for i in 1:length(dims)
		push!(n.a,   Vector{Float64}(undef, dims[i]))
		push!(n.∇a,  Vector{Float64}(undef, dims[i]))
	end

	for i in 2:length(dims)
		push!(n.z,   Vector{Float64}(undef, dims[i]))

		push!(n.w,   randn(dims[i], dims[i - 1]))
		push!(n.∇w,  Matrix{Float64}(undef, dims[i], dims[i - 1]))
		push!(n.Σ∇w, Matrix{Float64}(undef, dims[i], dims[i - 1]))

		push!(n.b,   randn(dims[i]))
		push!(n.∇b,  Vector{Float64}(undef, dims[i]))
		push!(n.Σ∇b, Vector{Float64}(undef, dims[i]))
	end

	return n
end

# leaky ReLU to avoid dead neurons
ReLU(x)  = max(0.01 * x, x)
ReLU′(x) = x >= 0 ? 1 : 0.01

σ(x)  = 1 / (1 + exp(-x))
σ′(x) = σ(x) * (1 - σ(x))

const act  = σ
const act′ = σ′

function forward!(n :: NN)
	for i in 1:length(n.dims) - 1
		n.z[i] = n.w[i] * n.a[i] + n.b[i]
		n.a[i + 1] = act.(n.z[i])
	end
end

cost(output, expected) = sum((output - expected) .^ 2)

function backprop!(n :: NN, expected :: Vector{Float64})
	len = length(n.dims)

	n.∇a[len] = 2 .* (n.a[len] - expected)

	for i in len - 1:-1:1
		n.∇a[i] .= 0
		n.∇b[i] = act′.(n.z[i]) .* n.∇a[i + 1]
		for j in 1:dims[i + 1]
			if i != 1
				n.∇a[i] += n.w[i][j, :] .* n.∇b[i][j]
			end
			n.∇w[i][j, :] = n.a[i] .* n.∇b[i][j]
		end
	end
end

# data: [(input, expected)], only one batch!
function train!(n :: NN, data :: Data) :: Float64
	for i in 1:length(n.dims) - 1
		n.Σ∇w[i] .= 0
		n.Σ∇b[i] .= 0
	end

	Σcost = 0
		
	for d in data
		n.a[1] = d[1]
		forward!(n)
		Σcost += cost(n.a[length(n.dims)], d[2]) / length(data)

		backprop!(n, d[2])
		n.Σ∇w .+= n.∇w / length(data)
		n.Σ∇b .+= n.∇b / length(data)
	end

	# play around with factor
	n.w -= 5 * n.Σ∇w
	n.b -= 5 * n.Σ∇b

	return Σcost
end
