using Test

using MicroGrad

@testset "Layers" begin
  @testset "Single" begin
    l = Layer(32, 8)
    @test length(params(l)) == 32 * 8 + 8
  end
  @testset "MultiLayerPerceptron" begin

    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    m = MultiLayerPerceptron(3, [4, 4, 1])
    nepochs = 200
    ϵ = 0.05
    train!(m, xs, ys; nepochs, ϵ)

  end
end
