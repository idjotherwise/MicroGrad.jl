module OperationsModule

using ..ValueModule

export inv, relu

function Base.:*(x::Value{T}, y::Value{T})::Value{T} where {T}
  out = Value(x.data * y.data, (x, y))
  function _backward()
    x.grad += y.data * out.grad
    y.grad += x.data * out.grad
  end
  out.backward = _backward
  out
end

function Base.:*(x, y::Value{T}) where {T}
  Base.:*(valueify(T, x), y)
end
function Base.:*(x::Value{T}, y) where {T}
  Base.:*(x, valueify(T, y))
end


function Base.:+(x::Value{T}, y::Value{T})::Value{T} where {T}
  out = Value(x.data + y.data, (x, y))
  function _backward()
    x.grad += one(T) * out.grad
    y.grad += one(T) * out.grad
  end
  out.backward = _backward
  out
end

function Base.:+(x, y::Value{T}) where {T}
  Base.:+(valueify(T, x), y)
end
function Base.:+(x::Value{T}, y) where {T}
  Base.:+(x, valueify(T, y))
end


function Base.:-(x::Value{T}, y::Value{T})::Value{T} where {T}
  out = Value(x.data - y.data, (x, y))
  function _backward()
    x.grad += one(T) * out.grad
    y.grad += one(T) * out.grad
  end
  out.backward = _backward
  out
end

function Base.:-(x, y::Value{T}) where {T}
  Base.:-(valueify(T, x), y)
end
function Base.:-(x::Value{T}, y) where {T}
  Base.:-(x, valueify(T, y))
end

function Base.:^(x::Value{T}, k::Number) where {T}
  out = Value(x.data^k, (x,))
  function _backward()
    x.grad += convert(T, k * x.data^(k - 1) * out.grad)
  end
  out.backward = _backward
  out
end

function Base.:^(x, y::Value{T}) where {T}
  Base.:^(valueify(T, x), y)
end
function Base.:^(x::Value{T}, y) where {T}
  Base.:^(x, valueify(T, y))
end

Base.inv(x::Value{T}) where {T} = ^(x, -1.0)

function Base.:/(x::Value{T}, y::Value{T}) where {T}
  c = x * ^(y, -1)
  c.children = Set([x, y])
  c
end

Base.:/(x, y::Value{T}) where {T} = valueify(T, x) * ^(y, -1)
Base.:/(x::Value{T}, y) where {T} = x * ^(valueify(T, y), -1)

function Base.tanh(x::Value)
  t = Base.tanh(x.data)
  out = Value(t, (x,))

  function _backward()
    x.grad += (1 - t^2) * out.grad
  end

  out.backward = _backward
  out
end

function relu(x::Value{T}) where {T}
  out = Value(max(zero(T), x.data), (x,))

  function _backward()
    x.grad += (x.data > 0) * out.grad
  end
  out.backward = _backward
  out
end

end