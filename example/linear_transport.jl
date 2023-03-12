module LinearTransport
using MPI
using MPIUtils
using HauntedArrays
using OrdinaryDiffEq
using DiffEqBase
using LinearAlgebra

const lx = 1.0
const nx = 3 # on each process
const c = 1.0

# function DiffEqBase.recursive_length(A::HauntedArray{T,1}) where {T}
#     xlocal = length(owned_values(A))
#     MPI.Allreduce(xlocal, MPI.SUM, get_comm(A))
# end

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

mypart = rank + 1
Δx = lx / (np * nx - 1)

# if mypart == 1
#     lid2gid = collect(1:(nx + 1))
#     lid2part = ones(Int, nx + 1)
#     lid2part[end] = mypart + 1

# elseif mypart == np
#     lid2gid = collect((nx * rank):(nx * (rank + 1)))
#     lid2part = mypart .* ones(Int, nx + 1)
#     lid2part[1] = mypart - 1
# else
#     lid2gid = collect((nx * rank):(nx * (rank + 1) + 1))
#     lid2part = mypart .* ones(Int, nx + 2)
#     lid2part[1] = mypart - 1
#     lid2part[end] = mypart + 1
# end

lid2gid = collect((rank * nx + 1):((rank + 1) * nx))
lid2part = mypart .* ones(Int, nx)
if mypart != np
    append!(lid2gid, nx + 1)
    append!(lid2part, mypart + 1)
end
if mypart != 1
    prepend!(lid2gid, rank * nx)
    prepend!(lid2part, mypart - 1)
end

@one_at_a_time @show lid2gid
@one_at_a_time @show lid2part

# Allocate
q = HauntedVector(comm, lid2gid, lid2part)
dq = similar(q)
p = (c = c, Δx = Δx)

# Init
(mypart == 1) && (q[1] = 1.0)

function f!(dq, q, p, t)

    update_ghosts!(q)
    for i in owned_rows(q)
        (local_to_global(q, i) == 1) && continue # boundary condition
        dq[i] = -c / p.Δx * (q[i] - q[i - 1])
    end
end

tspan = (0.0, 2.0)
prob = ODEProblem(f!, q, tspan, p)
sol = solve(prob, Tsit5())
q = sol.u[end]


update_ghosts!(q)
@one_at_a_time @show owned_values(q)

MPI.Finalize()
end