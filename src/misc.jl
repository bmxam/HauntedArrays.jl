"""
nx : number of points on each partition
mypart : MPI.Comm_rank + 1
"""
function generate_1d_partitioning(
    nx,
    mypart,
    nprocs,
    shuffle = false,
    rng = Random.GLOBAL_RNG,
)
    rank = mypart - 1
    lid2gid = collect((rank * nx + 1):((rank + 1) * nx))
    lid2part = mypart .* ones(Int, nx)
    if mypart != nprocs
        append!(lid2gid, (rank + 1) * nx + 1)
        append!(lid2part, mypart + 1)
    end
    if mypart != 1
        prepend!(lid2gid, rank * nx)
        prepend!(lid2part, mypart - 1)
    end

    if shuffle
        I = randperm(rng, length(lid2gid))
        lid2gid .= lid2gid[I]
        lid2part .= lid2part[I]
    end

    return lid2gid, lid2part
end
