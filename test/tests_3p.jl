module test
using MPI
using HauntedArrays
using Random
using LinearSolve

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

rng = MersenneTwister(1234)

function test_gather_3p()
    if rank == 0
        lid2gid = [1, 2, 6]
        lid2part = [1, 1, 1]

    elseif rank == 1
        lid2gid = [2, 3, 5, 6]
        lid2part = [1, 2, 2, 1]

    elseif rank == 2
        lid2gid = [3, 4, 5]
        lid2part = [2, 3, 2]
    end

    # Test 1
    x = HauntedArray(comm, lid2gid, lid2part)
    x.array .= lid2gid
    y = gather(x)
    @only_root @show y == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Test 2
    n = length(x)
    A = similar(x, Char, (n, n))
    if rank == 0
        A[1, :] = ['a', 'b', 'c']
        A[2, :] = ['d', 'e', 'f']
        A[3, :] = ['g', 'h', 'i']
    elseif rank == 1
        A[1, :] = ['e', 'j', 'k', 'f']
        A[2, :] = ['l', 'm', 'n', 'o']
        A[3, :] = ['p', 'q', 'r', 's']
        A[4, :] = ['h', 't', 'u', 'i']
    else
        A[1, :] = ['m', 'v', 'n']
        A[2, :] = ['w', 'x', 'y']
        A[3, :] = ['q', 'z', 'r']
    end

    # @one_at_a_time display(A)

    B = gather(A)
    @only_root begin
        bool = Bool[]
        push!(bool, B[1, 1] == 'a')
        push!(bool, B[1, 2] == 'b')
        push!(bool, B[1, 6] == 'c')
        push!(bool, B[2, 1] == 'd')
        push!(bool, B[2, 2] == 'e')
        push!(bool, B[2, 3] == 'j')
        push!(bool, B[2, 5] == 'k')
        push!(bool, B[2, 6] == 'f')
        push!(bool, B[3, 2] == 'l')
        push!(bool, B[3, 3] == 'm')
        push!(bool, B[3, 4] == 'v')
        push!(bool, B[3, 5] == 'n')
        push!(bool, B[3, 6] == 'o')
        push!(bool, B[4, 3] == 'w')
        push!(bool, B[4, 4] == 'x')
        push!(bool, B[4, 5] == 'y')
        push!(bool, B[5, 2] == 'p')
        push!(bool, B[5, 3] == 'q')
        push!(bool, B[5, 4] == 'z')
        push!(bool, B[5, 5] == 'r')
        push!(bool, B[5, 6] == 's')
        push!(bool, B[6, 1] == 'g')
        push!(bool, B[6, 2] == 'h')
        push!(bool, B[6, 3] == 't')
        push!(bool, B[6, 5] == 'u')
        push!(bool, B[6, 6] == 'i')
        @show all(bool)
    end


    @only_root println("End of test_gather_3p")
end

function test_ldiv_3p()
    if rank == 0
        lid2gid = [1, 2, 6]
        lid2part = [1, 1, 1]

    elseif rank == 1
        lid2gid = [2, 3, 5, 6]
        lid2part = [1, 2, 2, 1]

    elseif rank == 2
        lid2gid = [3, 4, 5]
        lid2part = [2, 3, 2]
    end

    Ag = rand(rng, 6, 6)
    Ag[1, 3:5] .= 0.0
    Ag[2, 4] = 0.0
    Ag[3, 1] = 0.0
    Ag[4, 1] = 0.0
    Ag[4, 2] = 0.0
    Ag[4, 6] = 0.0
    Ag[5, 1] = 0.0
    Ag[6, 4] = 0.0
    bg = rand(rng, 6)

    # @only_root @show Ag \ bg

    bl = HauntedArray(comm, lid2gid, lid2part)
    for lid in CartesianIndices(bl.lid2gid)
        bl[lid] = bg[bl.lid2gid[lid]]
    end

    Al = similar(bl, length(bl), length(bl))
    for lid in CartesianIndices(Al.lid2gid)
        # Al[lid]
        Al[lid] = Ag[Al.lid2gid[lid]]
    end

    @only_root display(Ag \ bg)
    cg = gather(Al \ bl)
    @only_root display(cg)

    error("ldiv not implemented yet")
end

test_gather_3p()
# test_ldiv_3p()


isinteractive() || MPI.Finalize()

end