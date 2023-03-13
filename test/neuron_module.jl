using Test

using MicroGrad

@testset "nnModule" begin
  n = Neuron(10)
  @test length(weight(n)) == 10

  @test typeof(params(n)) == Vector{Value{Float64,Set{Union{}},Function}}
end
