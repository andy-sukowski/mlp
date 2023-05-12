# See LICENSE file for copyright and license details.

# fill vectors and matrices
function init(dims, z, a, w, b, ∇a, ∇w, ∇b, Σ∇w, Σ∇b)
	len = length(dims)

	for i in 1:len
		push!( a, Vector{Float64}(undef, dims[i]))
		push!(∇a, Vector{Float64}(undef, dims[i]))
	end

	for i in 2:len
		push!( z,  Vector{Float64}(undef, dims[i]))

		push!( w,  randn(dims[i], dims[i - 1]))
		push!(∇w,  Matrix{Float64}(undef, dims[i], dims[i - 1]))
		push!(Σ∇w, Matrix{Float64}(undef, dims[i], dims[i - 1]))

		push!( b,  randn(dims[i]))
		push!(∇b,  Vector{Float64}(undef, dims[i]))
		push!(Σ∇b, Vector{Float64}(undef, dims[i]))
	end
end

# leaky ReLU to avoid dead neurons
ReLU(x)  = max(0.01 * x, x)
ReLU′(x) = x >= 0 ? 1 : 0.01

σ(x)  = 1 / (1 + exp(-x))
σ′(x) = σ(x) * (1 - σ(x))

const act  = σ
const act′ = σ′

function forward(len, z, a, w, b)
	for i in 1:len - 1
		z[i] = w[i] * a[i] + b[i]
		a[i + 1] = act.(z[i])
	end
end

cost(output, expected) = sum((output - expected) .^ 2)

function backprop(dims, expected, z, a, w, ∇a, ∇w, ∇b)
	len = length(dims)

	∇a[len] = 2 .* (a[len] - expected)

	for i in len - 1:-1:1
		∇a[i] .= 0
		∇b[i] = act′.(z[i]) .* ∇a[i + 1]
		for j in 1:dims[i + 1]
			if i != 1
				∇a[i] += w[i][j, :] .* ∇b[i][j]
			end
			∇w[i][j, :] = a[i] .* ∇b[i][j]
		end
	end
end

# data: [(input, expected)], only one batch!
function train(dims, data, z, a, w, b, ∇a, ∇w, ∇b, Σ∇w, Σ∇b)
	for i in 1:len - 1
		Σ∇w[i] .= 0
		Σ∇b[i] .= 0
	end

	Σcost = 0
		
	for d in data
		a[1] = d[1]
		forward(length(dims), z, a, w, b)
		Σcost += cost(a[length(dims)], d[2]) / length(data)

		backprop(dims, d[2], z, a, w, ∇a, ∇w, ∇b)
		Σ∇w .+= ∇w / length(data)
		Σ∇b .+= ∇b / length(data)
	end

	# play around with factor
	w .-= 5 * Σ∇w
	b .-= 5 * Σ∇b

	return Σcost
end
