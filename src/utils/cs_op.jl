export CSop

mutable struct CSop
    A::AbstractArray{Float32,2}
    nx::Int64
    ny::Int64
    nc::Int64
    s::Float32
end

function CSop(nc::Int; s::Float32 = 3.0f0)

    A = randn(Float32, nc, nc) + s * I
    A = A / opnorm(A)

    return CSop(A, 1, 1, nc, s)
end


function CSop(A::AbstractArray{Float32,2}, nc::Int; s::Float32 = 3.0f0)
    return CSop(A, 1, 1, nc, s)
end


function adjoint(A::CSop)
    return CSop(adjoint(A.A), A.nc; s = A.s)
end


function -(A::CSop)
    return CSop(-A.A, A.nc; s = A.s)
end


function *(CS::CSop, X::AbstractArray{Float32,4})
    nb = size(X, 4)
    return reshape(CS.A * reshape(X, :, nb), CS.nx, CS.ny, CS.nc, :)
end
