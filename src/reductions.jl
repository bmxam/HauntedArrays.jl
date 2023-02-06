# Inspired from PencilArrays : https://github.com/jipolanco/PencilArrays.jl
# https://raw.githubusercontent.com/jipolanco/PencilArrays.jl/master/src/reductions.jl

# We force specialisation on each function to avoid (tiny) allocations.
#
# Note that, for mapreduce, we can assume that the operation is commutative,
# which allows MPI to freely reorder operations.
#
# We also define mapfoldl (and mapfoldr) for completeness, even though the global
# operations are not strictly performed from left to right (or from right to
# left), since each process locally reduces first.
for (func, commutative) in [:mapreduce => true, :mapfoldl => false, :mapfoldr => false]
    @eval function Base.$func(
        f::F,
        op::OP,
        u::HauntedArray,
        etc::Vararg{HauntedArray};
        kws...,
    ) where {F,OP}
        foreach(v -> _check_compatible_arrays(u, v), etc)
        comm = get_comm(u)
        ups = map(owned_values, (u, etc...))
        rlocal = $func(f, op, ups...; kws...)
        op_mpi = MPI.Op(op, typeof(rlocal); iscommutative = $commutative)
        return MPI.Allreduce(rlocal, op_mpi, comm)
    end

    # # Make things work with zip(u::PencilArray, v::PencilArray, ...)
    # @eval function Base.$func(
    #     f::F,
    #     op::OP,
    #     z::Iterators.Zip{<:Tuple{Vararg{PencilArray}}};
    #     kws...,
    # ) where {F,OP}
    #     g(args...) = f(args)
    #     $func(g, op, z.is...; kws...)
    # end
end

function Base.:(==)(a::HauntedArray, b::HauntedArray)
    xlocal = ==(owned_values(a), owned_values(b))
    MPI.Allreduce(xlocal, &, get_comm(a))
end

function Base.any(f::F, u::HauntedArray) where {F<:Function}
    xlocal = any(f, owned_values(u))::Bool
    MPI.Allreduce(xlocal, |, get_comm(u))
end

function Base.all(f::F, u::HauntedArray) where {F<:Function}
    xlocal = all(f, owned_values(u))::Bool
    MPI.Allreduce(xlocal, &, get_comm(u))
end
