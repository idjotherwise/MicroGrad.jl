module TrainModule


using ..LayerModule
using ..ValueModule
export train!

mse(ys, ỹs) = sum([(y - ỹ)^2 for (y, ỹ) in zip(ys, ỹs)])

function train!(
  M::MultiLayerPerceptron,
  xs::Vector{Vector{Float64}},
  ys::Vector{Float64};
  nepochs::Int64 = 5,
  ϵ::Float64 = 0.05,
  loss_func::Function = mse,
)
  for e in 1:nepochs
    # forward pass
    y_pred = [M(x) for x in xs]
    loss = loss_func(ys, y_pred)

    # backward pass
    zero_gradients!(M)
    backward!(loss)
    # update
    for p in params(M)
      p.data += ϵ * grad(p)
    end

    new_y_pred = [M(x) for x in xs]
    new_loss = loss_func(ys, new_y_pred)
    if e % 50 == 0
      @show e, data(loss)
    end
  end
end


end