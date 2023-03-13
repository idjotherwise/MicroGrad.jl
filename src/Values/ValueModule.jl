module ValueModule

export Value,
  backward!,
  backward,
  topological_sort,
  children,
  data,
  grad,
  valueify

mutable struct Value{T,C,B}
  data::T
  grad::T
  children::C
  backward::B

  function Value(data, children)
    children = Set(children)
    D = typeof(data)
    C = typeof(children)
    return new{D,C,Function}(data, zero(D), children, () -> nothing)
  end
end
Value(data) = Value(data, ())

children(v::Value) = v.children
data(v::Value) = v.data
grad(v::Value) = v.grad
backward(v::Value) = v.backward

Base.in(x::Value{T}, y::Value{T}) where {T} = Base.in(x, children(y))

Base.show(io::IO, v::Value) = print(io, "Value($(data(v)), grad=$(grad(v)))")

function Base.show(io::IO, ::MIME"text/plain", v::Value)
  print(io, "Value($(v.data), grad=$(v.grad))")
end

valueify(T, x) = x isa Value ? x : Value(convert(T, x))

function topological_sort(v, topo = [])
  visited = Set()
  if v ∉ visited
    for child in children(v)
      topological_sort(child, topo)
    end
    push!(topo, v)
  end
  topo
end

function backward!(v::Value{T}) where {T}
  v.grad = one(T)
  for node in reverse(topological_sort(v))
    backward(node)()
  end
end

end