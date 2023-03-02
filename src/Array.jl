abstract type AbstractHauntedArray{T,N} <: AbstractArray{T,N} end

"""
`I` maybe be `CartesianIndex` for N > 1, or `Int` for vectors

# Dev notes
Storing `lid2gid` as `Array{CartesianIndex{N},N}` is a bit stupid since in practice the numering
is cartesian, hence we only need to store the global number of rows (and cols) and any element
can then be obtained combining these two infos.
"""
struct HauntedArray{T,N,E,I} <: AbstractHauntedArray{T,N} where {E<:AbstractExchanger}
    # The complete array on the current rank, including ghosts
    array::Array{T,N}

    # View of the vector `array` without the ghost, i.e only the values owned by this rank
    ownedValues

    # Structure to enable exchanging ghost values
    exchanger::E

    # Local to global index
    lid2gid::Array{I,N}

    # Local index to partition owning the element
    lid2part::Array{Int,N}

    # Local element indices, in `array`, that are owned by this rank -> `parentindices(ownedValues)`
    oids::Vector{I}

    # Local element indices, in `array`, that are ghosts
    ghids::Vector{I}

    HauntedArray(a::AbstractArray{T,N}, o, ex, l2g, l2p, oids, ghids) where {T,N} =
        new{T,N,typeof(ex),eltype(l2g)}(a, o, ex, l2g, l2p, oids, ghids)
end

@inline get_exchanger(A::HauntedArray) = A.exchanger
@inline get_comm(A::HauntedArray) = get_comm(get_exchanger(A))
@inline owned_values(A::HauntedArray) = A.ownedValues
@inline owned_indices(A::HauntedArray) = A.oids


function HauntedArray(
    comm::MPI.Comm,
    lid2gid::Array{I,N},
    lid2part::Array{Int,N},
    T = Float64,
) where {I,N}
    exchanger = MPIExchanger(comm, lid2gid, lid2part)

    # Additionnal infos
    mypart = MPI.Comm_rank(get_comm(exchanger)) + 1
    oids = findall(part -> part == mypart, lid2part)
    ghids = findall(part -> part != mypart, lid2part)

    return HauntedArray(exchanger, lid2gid, lid2part, oids, ghids, T)
end

function HauntedArray(
    exchanger::AbstractExchanger,
    lid2gid::Array{I,N},
    lid2part::Array{Int,N},
    oids,
    ghids,
    T = Float64,
) where {I,N}
    # Array with ghosts
    array = try
        zeros(T, size(lid2gid))
    catch e
        Array{T}(undef, size(lid2gid))
    end

    # Array without ghosts
    ownedValues = view(array, oids)

    return HauntedArray(array, ownedValues, exchanger, lid2gid, lid2part, oids, ghids)
end

const HauntedVector{T} = HauntedArray{T,1}
const HauntedMatrix{T} = HauntedArray{T,2}