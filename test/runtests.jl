using Test
using MicroGrad

TEST_FILES = [
  "operations_module.jl",
  "value_module.jl",
  "neuron_module.jl",
  "layer_module.jl",
]

@testset "MicroGrad.jl" begin
  for t in TEST_FILES
    @info "Testing $t..."
    path = joinpath(@__DIR__, t)
    @eval @time @testset $t begin
      include($path)
    end
  end
end
