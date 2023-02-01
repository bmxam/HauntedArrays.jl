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
    arrayNoGhost::Any

    lid2gid::Any # local (with ghosts) to global ids

    withGhosts::Vector{Bool}

    exchanger::Any
end

"""
Constructor for MPI backend

`lid2part` is 1-based, i.e the rank "0" corresponds to part "1"
"""
function HauntedVector(comm::MPI.Comm, lid2gid, lid2part, T = Float64)
    # Create array with ghosts
    array = zeros(T, length(lid2gid))

    # Create array without ghosts
    mypart = MPI.Comm_rank(comm) + 1
    myids = findall(lid -> lid2part[lid] == mypart, lid2part)
    arrayNoGhost = view(array, myids)

    exchanger = MPIExchanger(comm, lid2gid, lid2part)

    return HauntedVector{T}(array, arrayNoGhost, lid2gid, [true], exchanger)
end

@inline get_exchanger(v::HauntedVector) = v.exchanger

@inline set_with_ghosts(v::HauntedVector, val::Bool) = v.withGhosts[1] = val
@inline enable_ghosts(v::HauntedVector) = set_with_ghosts(v, true)
@inline disable_ghosts(v::HauntedVector) = set_with_ghosts(v, false)

@inline with_ghosts(v::HauntedVector) = v.withGhosts[1]
@inline get_array_with_ghosts(v::HauntedVector) = v.array
@inline get_array_without_ghosts(v::HauntedVector) = v.arrayNoGhost
@inline get_selected_array(v::HauntedVector) =
    with_ghosts(v) ? get_array_with_ghosts(v) : get_array_without_ghosts(v)

update_ghosts!(v::HauntedVector) =
    update_ghosts!(get_array_with_ghosts(v), get_exchanger(v))

Base.size(A::HauntedVector) = size(get_selected_array(A))
Base.getindex(A::HauntedVector, i::Int) = getindex(get_selected_array(A), i)
Base.setindex!(A::HauntedVector, v, i::Int) = setindex!(get_selected_array(A), v, i)