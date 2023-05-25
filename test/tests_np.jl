module test
using MPI
using MPIUtils
using Bcube
using BcubeParallel
using HauntedArrays
using Random
using LinearSolve

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np = MPI.Comm_size(comm)

rng = MersenneTwister(1234)

out_dir = joinpath(@__DIR__, "../myout")
tmp_path = joinpath(out_dir, "tmp.msh")

@only_root begin
    gen_rectangle_mesh(
        tmp_path,
        :tri;
        nx = 4,
        ny = 3,
        npartitions = np,
        split_files = true,
        create_ghosts = true,
    )
end
MPI.Barrier(comm)
dmesh = read_partitioned_msh(tmp_path, comm)
fSpace = FunctionSpace(:Lagrange, 1)
feSpace = Bcube.SingleFESpace(fSpace, parent(dmesh))
lid2gid, lid2part =
    BcubeParallel.compute_dof_global_numbering(Bcube._get_dhl(feSpace), dmesh)

function test_gather_np()
    # Vector
    Al = HauntedVector(comm, lid2gid, lid2part)
    nl = n_local_rows(Al)
    no = n_own_rows(Al)
    ng = MPI.Allreduce(no, MPI.SUM, comm)
    _Ag = rand(rng, ng)

    Al .= _Ag[lid2gid]

    Ag = gather(Al)

    @only_root begin
        @show Ag == _Ag
    end

    # Matrix
    Al = similar(Al, nl, nl)
    __Ag = rand(rng, ng, ng)
    for (i, j) in Iterators.product(1:nl, 1:nl)
        Al[i, j] = __Ag[local_to_global(Al, i), local_to_global(Al, j)]
    end

    Ag = gather(Al)
    l2g = HauntedArrays.gather_lid2gid(Al)
    @only_root begin
        _Ag = zeros(ng, ng)
        _Ag[l2g] .= __Ag[l2g]
        @show _Ag == Ag
    end

    @only_root println("End of test_gather_np")
end

test_ldiv_np() = error("ldiv not implemented yet")

function test_mul_np()
    α = 100.0

    xl = HauntedVector(comm, lid2gid, lid2part)
    nl = n_local_rows(xl)
    no = n_own_rows(xl)
    ng = MPI.Allreduce(no, MPI.SUM, comm)
    _xg = α .* rand(rng, ng)
    xl .= _xg[lid2gid]

    Al = similar(xl, nl, nl)
    __Ag = α .* rand(rng, ng, ng)
    for (i, j) in Iterators.product(1:nl, 1:nl)
        Al[i, j] = __Ag[local_to_global(Al, i), local_to_global(Al, j)]
    end

    l2g = HauntedArrays.gather_lid2gid(Al)
    @only_root begin
        _Ag = zeros(ng, ng)
        _Ag[l2g] .= __Ag[l2g]
    end

    bg = gather(Al * xl)

    @only_root begin
        _bg = _Ag * _xg

        for rtol in [1e-20 * 10^n for n = 0:10]
            @show rtol, all(isapprox.(_bg, bg; rtol = rtol))
        end
    end

    @only_root println("End of test_mul_np")
end

test_gather_np()
test_mul_np()
# test_ldiv_np()


isinteractive() || MPI.Finalize()

end