"""
Run on 2 procs only
"""
module test
using MPI
using MPIUtils
using HauntedArrays
using ForwardDiff

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

@assert np == 2 "example only for 2 procs"

mypart = rank + 1

if mypart == 1
    lid2gid = [1, 2, 3]
    lid2part = [1, 1, 2]
    function fl(x)
        y = similar(x)
        y[own_to_local(y)] .= [x[1] + 2 * x[2], x[1] + 2 * x[2] + 4 * x[3]]
        return y
    end
else
    lid2gid = [2, 3, 4]
    lid2part = [1, 2, 2]
    function fl(x)
        y = similar(x)
        y[own_to_local(y)] .= [x[1] + 2 * x[2] + 4x[3], x[2] + 2 * x[3]]
        return y
    end
end

xl = HauntedVector(comm, lid2gid, lid2part)
xl .= [1.0, 1.0, 1.0]
Jl = ForwardDiff.jacobian(fl, xl)
Jg = gather(Jl)


@only_root begin
    """
    f([x1, x2, x3, x4]) = [
    x1 + 2x2
    x1 + 2x2 + 4x3
    x2 + 2x3 + 4x4
    x3 + 2x4
    ]
    """
    function _fg(x)
        [
            x[1] + 2 * x[2],
            x[1] + 2 * x[2] + 4 * x[3],
            x[2] + 2 * x[3] + 4x[4],
            x[3] + 2 * x[4],
        ]
    end

    _Jg = ForwardDiff.jacobian(_fg, [1.0, 1.0, 1.0, 1.0])
    display(_Jg)
    display(Jg)
end

isinteractive() || MPI.Finalize()
end