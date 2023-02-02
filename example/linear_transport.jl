module LinearTransport
using MPI
using HauntedArrays
using DifferentialEquations

const lx = 1.0
const nx = 5 # on each process
const c = 1.0

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
q = HauntedVector(comm, lid2gid, lid2part)
dq = HauntedVector(comm, lid2gid, lid2part)
p = (c = c, Δx = Δx)

# Init
enable_ghosts(q)
(mypart == 1) && (q[1] = 1.0)

function f!(dq, q, p, t)
    @one_at_a_time println("proc $rank is waiting at Barrier 7")
    MPI.Barrier(comm)

    update_ghosts!(q)
    enable_ghosts.((dq, q))
    for i = 2:length(q)
        dq[i] = -c / p.Δx * (q[i] - q[i-1])
    end
    disable_ghosts.((dq, q))

    @one_at_a_time println("proc $rank is waiting at Barrier 8")
    MPI.Barrier(comm)
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
    disable_ghosts(q) # mandatory ! (don't know why though...)
    tspan = (0.0, 1.0)
    prob = ODEProblem(f!, q, tspan, p)
    sol = solve(prob, Tsit5())
end

q = sol.u[end]

@one_at_a_time println("proc $rank is waiting at Barrier 9")
MPI.Barrier(comm)

@only_root println("without ghosts (sync)")
update_ghosts!(q)
disable_ghosts(q)
@one_at_a_time @show q

MPI.Finalize()
end