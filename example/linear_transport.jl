module LinearTransport
using MPI
using HauntedArrays
using OrdinaryDiffEq
using DiffEqBase
using LinearAlgebra

const lx = 1.0
const nx = 3 # on each process
const c = 1.0

# LinearAlgebra.norm(A::HauntedArray{T,1}, p::Real = 2) where {T} = mynorm(A, p)
# LinearAlgebra.norm(A::HauntedArray{T,1}, p::Real = 2) where {T} = mynorm(A, p)

mynorm(A, p::Real = 2) = LinearAlgebra.norm(A, p)

function mynorm(A::HauntedArray{T,1}, p::Real = 2) where {T}
    @assert p == 2 "p # 2 : p = $p"
    a = owned_values(A)
    xlocal = sum(a .* a)
    println("hello")
    res = MPI.Allreduce(xlocal, MPI.SUM, get_comm(A))
    return √(res)
end

Base.maximum(A::HauntedArray) = error("not supported")
Base.minimum(A::HauntedArray) = error("not supported")
Base.extrema(A::HauntedArray) = error("not supported")
Base.similar(A::HauntedArray, dims::Dims) = error("not supported")
Base.similar(A::HauntedArray, ::Type{S}, dims::Dims) where {S} = error("not supported")

function DiffEqBase.recursive_length(A::HauntedArray{T,1}) where {T}
    xlocal = length(owned_values(A))
    MPI.Allreduce(xlocal, MPI.SUM, get_comm(A))
end



MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

mypart = rank + 1
Δx = lx / (np * nx - 1)

lid2gid = collect(1:nx)
lid2part = mypart .* ones(Int, nx)
if mypart > 1
    # Shift numbering
    lid2gid .+= rank * nx

    # Add ghost (last cell from prev proc)
    prepend!(lid2gid, rank * nx)
    prepend!(lid2part, rank)
end

@one_at_a_time @show lid2part

# Allocate
q = HauntedArray(comm, lid2gid, lid2part)
dq = HauntedArray(comm, lid2gid, lid2part)
p = (c = c, Δx = Δx)

# Init
(mypart == 1) && (q[1] = 1.0)

function f!(dq, q, p, t)

    update_ghosts!(q)
    for i = 2:length(q)
        dq[i] = -c / p.Δx * (q[i] - q[i-1])
    end
end

# Explicit time integration
if false
    nite = 10
    Δt = Δx / c
    for i = 1:nite
        f!(dq, q, p, 0.0)
        @. q += Δt * dq
    end
end

# Diffeq
if true
    tspan = (0.0, 2.0)
    prob = ODEProblem(f!, q, tspan, p)
    sol = solve(prob, Tsit5())
    # sol = solve(prob, Tsit5(), maxiters = 2, internalnorm = mynorm)
end

println("proc $rank is waiting at Barrier 8")
MPI.Barrier(comm)
q = sol.u[end]

println("proc $rank is waiting at Barrier 9")
MPI.Barrier(comm)

update_ghosts!(q)
@one_at_a_time @show owned_values(q)

MPI.Finalize()
end