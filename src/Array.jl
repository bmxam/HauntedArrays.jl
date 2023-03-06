abstract type AbstractHauntedArray{T,N} <: AbstractArray{T,N} end

"""
When N > 1, the HauntedArray has the same size along all dimensions (square matrix for instance).
Then `lid2gid` is assumed to be the same along all dimensions.

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
    lid2part::Vector{Int}

    # Local element indices, in the first dimension of `array`, that are owned by this rank
    oids::Vector{I}

    function HauntedArray(a::AbstractArray{T,N}, ex, l2g, l2p, oids) where {T,N}
        new{T,N,typeof(ex),eltype(l2g)}(a, ex, l2g, l2p, oids)
    end
end

@inline get_exchanger(A::HauntedArray) = A.exchanger
@inline get_comm(A::HauntedArray) = get_comm(get_exchanger(A))
@inline owned_indices(A::HauntedArray) = A.oids
@inline owned_values(A::HauntedArray) = view(A, ntuple(d -> owned_indices(A), ndims(A)))


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
    ghids = findall(part -> part != mypart, lid2part)

    return HauntedArray(exchanger, lid2gid, lid2part, oids, ghids, ndims, T)
end

function HauntedArray(
    exchanger::AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oids,
    ghids,
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

    return HauntedArray(array, exchanger, lid2gid, lid2part, oids, ghids)
end

function HauntedVector(
    comm::MPI.Comm,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    T = Float64,
) where {I}
    HauntedArray(comm, lid2gid, lid2part, 1, T)
end