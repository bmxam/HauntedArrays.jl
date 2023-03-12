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
  - `lid2part` : Local row index to partition owning the element
  - `oids` : rows indices that are owned by the current partition

# Warning

For now, the `exchanger` is only relevant for HauntedVector.
"""
struct HauntedArray{T,N,E,I} <:
       AbstractHauntedArray{T,N} where {E<:AbstractExchanger,I<:Integer}
    # The complete array on the current rank, including ghosts
    array::Array{T,N}

    # Structure to enable exchanging ghost values
    exchanger::E

    # Local to global index. For N > 1, `lid2gid` is shared by each dimension
    lid2gid::Vector{I}

    # Local index to partition owning the element
    # -> will most likely disappear
    lid2part::Vector{Int}

    # Local element indices, in the first dimension of `array`, that are owned by this rank
    oids::Vector{I}

    function HauntedArray(a::AbstractArray{T,N}, ex, l2g, l2p, oids) where {T,N}
        new{T,N,typeof(ex),eltype(l2g)}(a, ex, l2g, l2p, oids)
    end
end

@inline get_exchanger(A::HauntedArray) = A.exchanger
@inline get_comm(A::HauntedArray) = get_comm(get_exchanger(A))
@inline local_to_global(A::HauntedArray) = A.lid2gid
@inline local_to_part(A::HauntedArray) = A.lid2part

"""
Return an array of the "rows" (i.e first dimension of the local array) that are truly owned
by the current partition.
"""
@inline owned_rows(A::HauntedArray) = A.oids

"""
Create a tuple with indices, in each array dimension, that are owned by the current partition

# Warning : misleading for SparseArrays
"""
function _owned_indices(A::HauntedArray)
    return ntuple(d -> (d == 1) ? owned_rows(A) : collect(1:size(A, d)), ndims(A))
end

"""
Build a view of the parent array, with the elements that are owned by the current partition

# Warning : misleading for SparseArrays
"""
@inline owned_values(A::HauntedArray) = view(parent(A), owned_indices(A))

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
) where {I}
    @assert ndims <= 2 "`ndims > 2 is not yet supported"

    exchanger = MPIExchanger(comm, lid2gid, lid2part)

    # Additionnal infos
    mypart = MPI.Comm_rank(get_comm(exchanger)) + 1
    oids = findall(part -> part == mypart, lid2part)

    return HauntedArray(exchanger, lid2gid, lid2part, oids, ndims, T)
end

function HauntedArray(
    exchanger::AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oids,
    ndims::Int,
    T = Float64,
) where {I}
    n = length(lid2gid)
    dims = ntuple(i -> n, ndims)

    # Array with ghosts
    array = try
        zeros(T, dims)
    catch e
        Array{T}(undef, dims)
    end

    return HauntedArray(array, exchanger, lid2gid, lid2part, oids)
end

function HauntedVector(
    comm::MPI.Comm,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    T = Float64,
) where {I}
    HauntedArray(comm, lid2gid, lid2part, 1, T)
end