abstract type AbstractHauntedArray{T,N} <: AbstractArray{T,N} end

"""
When N > 1, the HauntedArray has the same size along all dimensions (square matrix for instance).
Then `lid2gid` is assumed to be the same along all dimensions.

For now, only the first dimension of the array is "distributed". For instance for a matrix,
each part owns a certain number of rows (but all the columns for these rows).

# Properties

  - `array` : In a future version, `array` will be of type `A <: AbstractArray` to allow SparseArrays
  - `exchanger` : For now, only relevant for HauntedVector
  - `lid2gid` : local row index to global row index
  - `lid2part` : Local row index to partition owning the element. Will most likely disappear. For now,
    only serves to create a Matrix from a Vector
  - `oid2lid` : rows indices that are owned by the current partition

# Warning

For now, the `exchanger` is only relevant for HauntedVector.
"""
struct HauntedArray{T,N,A,E,I,C} <: AbstractHauntedArray{T,N}
    # The complete array on the current rank, including ghosts
    array::A

    # Structure to enable exchanging ghost values
    exchanger::E

    # Local to global index. For N > 1, `lid2gid` is shared by each dimension
    lid2gid::Vector{I}

    # Local index to partition owning the element
    lid2part::Vector{Int}

    # Own to local element indices, in the first dimension of `array`, that are owned by this rank
    oid2lid::Vector{I}

    # Cache
    cache::C

    function HauntedArray(
        a::AbstractArray{T,N},
        ex::AbstractExchanger,
        l2g::Vector{I},
        l2p::Vector{Int},
        o2l::Vector{I},
        c::AbstractCache,
    ) where {T,N,I<:Integer}
        new{T,N,typeof(a),typeof(ex),I,typeof(c)}(a, ex, l2g, l2p, o2l, c)
    end
end

@inline get_exchanger(A::HauntedArray) = A.exchanger
@inline get_comm(A::HauntedArray) = get_comm(get_exchanger(A))
@inline local_to_global(A::HauntedArray) = A.lid2gid
@inline local_to_global(A::HauntedArray, i) = A.lid2gid[i]
@inline own_to_local(A::HauntedArray) = A.oid2lid
@inline own_to_local(A::HauntedArray, i) = A.oid2lid[i]
@inline own_to_global(A::HauntedArray) = A.lid2gid[A.oid2lid]
@inline own_to_global(A::HauntedArray, i) = A.lid2gid[A.oid2lid[i]]
@inline local_to_part(A::HauntedArray) = A.lid2part
@inline local_to_part(A::HauntedArray, i) = A.lid2part[i]
@inline n_local_rows(A::HauntedArray) = length(local_to_global(A))
@inline n_own_rows(A::HauntedArray) = length(own_to_local(A))
@inline get_cache(A::HauntedArray) = A.cache
@inline get_part(A::HauntedArray) = MPI.Comm_rank(get_comm(A)) + 1
@inline owned_by_me(A::HauntedArray, i) = local_to_part(A, i) == get_part(A)

"""
Return an array of the "rows" (i.e first dimension of the local array) that are truly owned
by the current partition.
"""
@inline own_to_local_rows(A::HauntedArray) = own_to_local(A)

"""
Create a tuple with local indices, in each array dimension, that are owned by the current partition

# Warning : misleading for SparseArrays
"""
function _own_to_local_ndims(A::HauntedArray)
    return ntuple(d -> (d == 1) ? own_to_local_rows(A) : collect(1:size(A, d)), ndims(A))
end

"""
Create a tuple with global indices, in each array dimension, that are owned by the current partition

# Warning : misleading for SparseArrays
"""
function _own_to_global_ndims(A::HauntedArray)
    return ntuple(d -> (d == 1) ? own_to_global(A) : local_to_global(A), ndims(A))
end

"""
Build a view of the parent array, with the elements that are owned by the current partition

# Warning : misleading for SparseArrays
"""
@inline owned_values(A::HauntedArray) = view(parent(A), _own_to_local_ndims(A)...)

@inline set_owned_values(A::HauntedArray, B::AbstractArray) = owned_values(A) .= B

"""
Transform a `Vector` index into a Vector of nd-CartesianIndices
(assuming same length in all Array direction)
"""
# function _scalar_index_to_array_index(I::AbstractVector, nd)
#     tuple_I = ntuple(d -> I, nd)
#     return [CartesianIndex(i) for i in Iterators.product(tuple_I...)]
# end


const HauntedVector{T} = HauntedArray{T,1}
const HauntedMatrix{T} = HauntedArray{T,2}


function HauntedArray(
    comm::MPI.Comm,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    ndims::Int,
    T = Float64,
    C::Type{<:AbstractCache} = EmptyCache;
    kwargs...,
) where {I}
    @assert ndims <= 2 "`ndims > 2 is not yet supported"

    exchanger = MPIExchanger(comm, lid2gid, lid2part)

    # Additionnal infos
    mypart = MPI.Comm_rank(get_comm(exchanger)) + 1
    oid2lid = findall(part -> part == mypart, lid2part)

    return HauntedArray(exchanger, lid2gid, lid2part, oid2lid, ndims, T, C)
end

function HauntedArray(
    exchanger::AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oid2lid::Vector{I},
    ndims::Int,
    T = Float64,
    C::Type{<:AbstractCache} = EmptyCache;
    kwargs...,
) where {I}
    n = length(lid2gid)
    dims = ntuple(i -> n, ndims)

    # Array with ghosts
    array = try
        zeros(T, dims)
    catch e
        Array{T}(undef, dims)
    end

    return HauntedArray(array, exchanger, lid2gid, lid2part, oid2lid, C; kwargs)
end

function HauntedArray(
    array::AbstractArray{T,N},
    exchanger::AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oid2lid::Vector{I},
    C::Type{<:AbstractCache} = EmptyCache;
    kwargs...,
) where {T,N,I}
    # Build the cache
    cache = build_cache(C, array, exchanger, lid2gid, lid2part, oid2lid; kwargs)

    return HauntedArray(array, exchanger, lid2gid, lid2part, oid2lid, cache)
end

function HauntedVector(
    comm::MPI.Comm,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    T = Float64;
    cacheType::Type{<:AbstractCache} = EmptyCache,
    kwargs...,
) where {I}
    HauntedArray(comm, lid2gid, lid2part, 1, T, cacheType; kwargs)
end

"""
Build a HauntedMatrix with values of `parent_matrix` and exchanger obtained from the
HauntedVector `x`
"""
function HauntedMatrix(parent_matrix::AbstractArray, x::HauntedVector)
    return HauntedArray(
        parent_matrix,
        get_exchanger(x), # exchanger not relevant for HauntedMatrix
        local_to_global(x),
        local_to_part(x),
        own_to_local(x),
        typeof(get_cache(x)),
    )
end
