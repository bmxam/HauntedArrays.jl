module test
using MPI
using HauntedArrays
# using FastBroadcast

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    lid2gid = [1, 2, 3, 4]
    lid2part = [1, 1, 1, 2]
else
    lid2gid = [3, 4, 5, 6, 7]
    lid2part = [1, 2, 2, 2, 2]
end

v = HauntedVector(comm, lid2gid, lid2part)
enable_ghosts(v)
if rank == 0
    v .= [1, 2, 3, 4]
else
    v .= [11, 12, 13, 14, 15]
end

@only_root println("without ghosts")
disable_ghosts(v)
@one_at_a_time @show v

@only_root println("with ghosts (no sync)")
enable_ghosts(v)
@one_at_a_time @show v

@only_root println("with ghosts (sync)")
update_ghosts!(v)
@one_at_a_time @show v

@only_root @show similar(v)
@only_root @show typeof(v)

# disable_ghosts(v)
# enable_ghosts(v)
# @only_root @show A
@one_at_a_time @show axes(v)

A = similar(v, eltype(v), axes(v))
A .= 1.0

# tmp = @.. broadcast = false v / A

MPI.Finalize()
end