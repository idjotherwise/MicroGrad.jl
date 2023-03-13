module LayerModule

using ..NeuronModule

using Reexport

export Layer, neurons, input, MultiLayerPerceptron, layers, zero_gradients!

@reexport import ..NeuronModule: params

struct Layer{N}
  neurons::N

  function Layer(n_inputs, n_outputs, σ = tanh)
    neurons = [Neuron(n_inputs, σ) for _ in 1:n_outputs]
    N = typeof(neurons)
    new{N}(neurons)
  end
end

neurons(l::Layer) = l.neurons
Base.first(l::Layer) = first(neurons(l))
params(l::Layer) = vcat([params(n) for n in neurons(l)]...)

function (L::Layer)(x)
  out = [N(x) for N in neurons(L)]
  length(out) == 1 ? out[1] : out
end

input(l::Layer) = first(neurons(l))

function Base.show(io::IO, l::Layer)
  n_inputs = length(input(l))
  n_outputs = length(neurons(l))
  print(io, "Layer ($n_inputs inputs - $n_outputs outputs) with $(n_outputs) neurons of type: ")
  show(io, input(l))
end

struct MultiLayerPerceptron{L}
  layers::L

  function MultiLayerPerceptron(n_inputs, n_outputs, σ = tanh)
    layer_sizes = vcat(n_inputs, n_outputs)
    n_layers = length(layer_sizes)
    layers = [Layer(layer_sizes[i], layer_sizes[i + 1], i == n_layers - 1 ? identity : σ) for i in 1:(n_layers - 1)]
    L = typeof(layers)
    new{L}(layers)
  end
end

layers(m::MultiLayerPerceptron) = m.layers

function (M::MultiLayerPerceptron)(x)
  for L in layers(M)
    x = L(x)
  end
  x
end

params(M::MultiLayerPerceptron) = vcat([params(L) for L in layers(M)]...)

Base.first(M::MultiLayerPerceptron) = first(layers(M))
input(M::MultiLayerPerceptron) = first(M)
Base.last(M::MultiLayerPerceptron) = last(layers(m))

function Base.show(io::IO, M::MultiLayerPerceptron)
  n_inputs = M |> layers |> first |> neurons |> first |> weight |> length
  n_outputs = M |> layers |> last |> neurons |> length
  println(io, "MLP ($n_inputs inputs - $n_outputs outputs) with $(length(layers(M))) layers:")
  for (i, layer) in enumerate(layers(M))
    print(io, " Layer $i: ")
    show(io, layer)
    i != length(layers(M)) && print(io, "\n")
  end
end

function zero_gradients!(model::MultiLayerPerceptron)
  for p in params(model)
    p.grad = 0
  end
end

end