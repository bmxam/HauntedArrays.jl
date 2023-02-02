const AbstractHauntedVector{T} = AbstractHauntedArray{T,1}
const AbstractHauntedMatrix{T} = AbstractHauntedArray{T,2}

"""
# Dev notes
There are different possibilites for this structure:
    - one array attribute that include ghosts + a view on this array without the ghosts
    - one array attribute that include ghosts + an index of the elements that are no ghosts
    - one array without ghosts, one array only with the ghosts
    - ...
"""
struct HauntedVector{T} <: AbstractHauntedVector{T}
    # The complete array on the current rank, including ghosts
    array::Vector{T}

    # Currently : a view on the array with the ghosts
    # May be replaced by a list of indices
    arrayNoGhost

    lid2gid # local (with ghosts) to global ids

    ghostsEnabled::Vector{Bool}

    exchanger

    HauntedVector(array, arrayNoGhost, lid2gid, exchanger, ghostsEnabled = false) =
        new{eltype(array)}(array, arrayNoGhost, lid2gid, [ghostsEnabled], exchanger)
end

"""
Constructor for MPI backend

`lid2part` is 1-based, i.e the rank "0" corresponds to part "1"
"""
function HauntedVector(
    comm::MPI.Comm,
    lid2gid,
    lid2part,
    T = Float64;
    ghostsEnabled = false,
)
    exchanger = MPIExchanger(comm, lid2gid, lid2part)
    return HauntedVector(exchanger, lid2gid, lid2part, T; ghostsEnabled)
end

function HauntedVector(
    exchanger::MPIExchanger,
    lid2gid,
    lid2part,
    T = Float64;
    ghostsEnabled = false,
)
    # Create array with ghosts
    array = zeros(T, length(lid2gid))

    # Create array without ghosts
    mypart = MPI.Comm_rank(get_comm(exchanger)) + 1
    myids = findall(part -> part == mypart, lid2part)
    arrayNoGhost = view(array, myids)

    return HauntedVector(array, arrayNoGhost, lid2gid, exchanger, ghostsEnabled)
end

@inline get_comm(v::HauntedVector) = get_comm(get_exchanger(v))
@inline get_exchanger(v::HauntedVector) = v.exchanger
@inline get_lid2gid(v::HauntedVector) = v.lid2gid

@inline set_ghosts_enabled(v::HauntedVector, val::Bool) = v.ghostsEnabled[1] = val
@inline enable_ghosts(v::HauntedVector) = set_ghosts_enabled(v, true)
@inline disable_ghosts(v::HauntedVector) = set_ghosts_enabled(v, false)

@inline with_ghosts(v::HauntedVector) = v.ghostsEnabled[1]
@inline ghosts_enabled(v::HauntedVector) = with_ghosts(v)
@inline get_array_with_ghosts(v::HauntedVector) = v.array
@inline get_array_without_ghosts(v::HauntedVector) = v.arrayNoGhost
@inline get_selected_array(v::HauntedVector) =
    with_ghosts(v) ? get_array_with_ghosts(v) : get_array_without_ghosts(v)

update_ghosts!(v::HauntedVector) =
    update_ghosts!(get_array_with_ghosts(v), get_exchanger(v))

Base.size(A::HauntedVector) = size(get_selected_array(A))
Base.getindex(A::HauntedVector, i::Int) = getindex(get_selected_array(A), i)
Base.setindex!(A::HauntedVector, v, i::Int) = setindex!(get_selected_array(A), v, i)
function Base.similar(A::HauntedVector, ::Type{S}) where {S}
    array = similar(get_array_with_ghosts(A), S)
    ind = parentindices(get_array_without_ghosts(A))[1] # `parentindices` returns a Tuple
    arrayNoGhost = view(array, ind)

    return HauntedVector(
        array,
        arrayNoGhost,
        get_lid2gid(A),
        get_exchanger(A),
        with_ghosts(A),
    )
end

Base.similar(A::HauntedVector{T}) where {T} = similar(A, T)

# Base.similar(A::HauntedVector, dims::Dims) = error("not supported yet")
# Base.similar(A::HauntedVector, ::Type{S}, dims::Dims) where {S} = error("not supported yet")
# Base.similar(A::HauntedVector, ::Type{S}, inds) where {S} = error("not supported yet")

function Base.zero(A::HauntedVector)
    B = similar(A)
    array = get_array_with_ghosts(B)
    array .= zero(array)
    return B
end


gather(v::HauntedVector, root = 0) = MPI.gather(get_array_without_ghosts(v), root, comm)