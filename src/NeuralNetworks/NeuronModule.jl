module NeuronModule

using ..ValueModule

export Neuron, σ, weight, bias, params

rand_unif(a::Number, b::Number) = a + (b - a) * rand()

struct Neuron{W,B,A}
  weight::W
  bias::B
  σ::A

  function Neuron(n, σ = tanh)
    w = [Value(rand_unif(-1, 1)) for _ in 1:n]
    b = Value(0.0)
    new{typeof(w),typeof(b),typeof(σ)}(w, b, σ)
  end
end

σ(n::Neuron) = n.σ
weight(n::Neuron) = n.weight
bias(n::Neuron) = n.bias

Base.length(n::Neuron) = Base.length(weight(n))

params(n::Neuron) = vcat(weight(n), bias(n))

function Base.show(io::IO, n::Neuron)
  print(io, "Neuron: $(length(n)) inputs -> 1 output ($(σ(n)))")
end

(N::Neuron)(x) = σ(N)(sum(weight(N) .* x) + bias(N))

end