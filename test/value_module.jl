using Test

using MicroGrad

@testset "Values" begin

  @testset "Gradient" begin
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b = Value(6.8813735870195432)

    c1 = x1 * w1
    c2 = x2 * w2
    d = c1 + c2
    n = d + b
    o = Base.tanh(n)
    backward!(o)

    @test grad(o) == 1.0
    @test grad(n) ≈ 0.5

    @test grad(d) ≈ 0.5
    @test grad(b) ≈ 0.5

    @test grad(c2) ≈ 0.5
    @test grad(c1) ≈ 0.5

    @test grad(x1) ≈ -1.5
    @test grad(w1) ≈ 1.0

    @test grad(x2) ≈ 0.5
    @test grad(w2) ≈ 0
  end
end