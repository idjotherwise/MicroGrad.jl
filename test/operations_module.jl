using Test
using MicroGrad

@testset "Operations on Values" begin

  function operation_interface(v1::Number, v2::Number, op::Function, k::Union{Number,Nothing} = nothing)
    a = Value(v1)
    b = Value(v2)
    c = op(a, b)

    @test data(c) == op(v1, v2)
    @test isempty(children(a))
    @test isempty(children(b))
    @test a ∈ c
    @test b ∈ c
  end

  @testset "Addition" begin
    operation_interface(2.5, -7.0, Base.:+)
  end
  @testset "Subtraction" begin
    operation_interface(2.5, -7.0, Base.:-)
  end

  @testset "Multiply" begin
    operation_interface(3 // 8, 1 // 8, Base.:*)
  end

  @testset "Divide" begin
    operation_interface(4.0, 2.0, Base.:/)
  end
  @testset "Exponentiating" begin
    a = Value(2.0)
    c = Base.:^(a, 2)
    @test data(c) == 4.0
    @test isempty(children(a))
  end

end