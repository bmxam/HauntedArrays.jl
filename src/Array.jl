abstract type AbstractHauntedArray{T,N} <: AbstractArray{T,N} end

"""
# Dev notes
Storing `lid2gid` as `Array{CartesianIndex{N},N}` is a bit stupid since in practice the numering
is cartesian, hence we only need to store the global number of rows (and cols) and any element
can then be obtained combining these two infos.
"""
struct HauntedArray{T,N,E} <: AbstractHauntedArray{T,N} where {E<:AbstractExchanger}
    # The complete array on the current rank, including ghosts
    array::Array{T,N}

    # View of the vector `array` without the ghost, i.e only the values owned by this rank
    ownedValues

    # Structure to enable exchanging ghost values
    exchanger::E

    # Local to global index
    lid2gid::Array{CartesianIndex{N},N}

    # Element indices, in `array`, that are owned by this rank -> `parentindices(ownedValues)`
    oids::Vector{CartesianIndex{N}}

    # Element indices, in `array`, that are ghosts
    ghids::Vector{CartesianIndex{N}}

    HauntedArray(a::AbstractArray{T,N}, o, ex, l2g, oids, ghids) where {T,N} =
        HauntedArray{T,N,typeof(ex)}(a, o, ex, l2g, oids, ghids)
end

@inline get_exchanger(A::HauntedArray) = A.exchanger
@inline get_comm(A::HauntedArray) = get_comm(get_exchanger(A))


function HauntedArray(
    comm::MPI.Comm,
    lid2gid::Array{CartesianIndex{N},N},
    lid2part::Array{Int,N},
    T = Float64,
) where {N}
    exchanger = MPIExchanger(comm, lid2gid, lid2part)

    # Array with ghosts
    array = zeros(T, size(lid2gid))

    # Create array without ghosts
    mypart = MPI.Comm_rank(get_comm(exchanger)) + 1
    oids = findall(part -> part == mypart, lid2part)
    ghids = findall(part -> part != mypart, lid2part)
    ownedValues = view(array, oids)

    return HauntedArray(array, ownedValues, exchanger, lid2gid, oids, ghids)
end

function HauntedVector(comm::MPI.Comm, lid2gid, lid2part, T = Float64)
    HauntedArray(comm, map(x -> CartesianIndex(x...), lid2gid), lid2part, T)
end