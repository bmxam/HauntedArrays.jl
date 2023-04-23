module test
using MPI
using HauntedArrays

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

@assert np == 1 "Tests to be ran on one proc only"

"""
* test of `similar` from vector -> matrix
* test of `mul!`
"""
function test_mul!_1p()
    lid2gid = [10, 12, 56]
    lid2part = [1, 1, 1]

    n = length(lid2part)

    # Test 1
    y = rand(n)
    B = rand(n, n)

    x = HauntedVector(comm, lid2gid, lid2part)
    A = similar(x, n, n)

    x.array .= y
    A.array .= B

    @show parent(A * x) == B * y

    # Test 2
    C = rand(n, n)
    D = similar(A)
    D.array .= C

    @show parent(A * D) == B * C
end

test_mul!_1p()




isinteractive() || MPI.Finalize()
end