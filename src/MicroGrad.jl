module MicroGrad

include("Values/ValueModule.jl")
include("Values/OperationsModule.jl")
include("NeuralNetworks/NeuronModule.jl")
include("NeuralNetworks/LayerModule.jl")
include("NeuralNetworks/TrainModule.jl")

using Reexport

@reexport using .ValueModule
@reexport using .OperationsModule
@reexport using .NeuronModule
@reexport using .LayerModule
@reexport using .TrainModule


end
