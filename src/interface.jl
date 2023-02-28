Base.parent(A::HauntedArray) = A.array

Base.size(A::HauntedArray) = size(parent(A))
# Base.getindex(A::HauntedArray, i::Int) = getindex(parent(A), i)
Base.setindex!(A::HauntedArray, v, i::Int) = setindex!(parent(A), v, i)

function Base.getindex(A::HauntedArray, i::Int)
    # println("getting element $i of rank $(MPI.Comm_rank(get_comm(A)))")
    getindex(parent(A), i)
end

function Base.getindex(A::HauntedArray, I::Vararg{Int,N}) where {N}
    # println("getting element $I of rank $(MPI.Comm_rank(get_comm(A)))")
    getindex(parent(A), I...)
end
# Base.getindex(A::HauntedArray, I...) = getindex(parent(A), I)

function Base.similar(A::HauntedArray, ::Type{S}) where {S}
    # Parent similar
    array = similar(parent(A), S)

    # Create array without ghosts
    ownedValues = view(array, A.oids)

    return HauntedArray(array, ownedValues, A.exchanger, A.lid2gid, A.oids, A.ghids)
end

Base.similar(A::HauntedArray{T}) where {T} = similar(A, T)

# Base.similar(A::HauntedArray, dims::Dims) = error("similar(A, dims::Dims), $dims")
function Base.similar(A::HauntedArray{T,N}, ::Type{S}, dims::Dims{N}) where {S,T,N}
    similar(A, S)
end

function Base.similar(A::HauntedVector, ::Type{S}, dims::Dims{2}) where {S}
    n = length(parent(A))

    # Matrix case
    @assert dims[1] == dims[2] "Only square Matrix supported for now"
    @assert dims[1] == n "Number of rows must match number of elts of the vector"

    lid2gid = [CartesianIndex(gi, gj) for gi in A.lid2gid, gj in A.lid2gid]

    lid2part = matrix_from_vector(get_exchanger(A), n, A.lid2part)

    return HauntedArray(get_comm(A), lid2gid, lid2part)
end

function Base.zero(A::HauntedArray)
    B = similar(A)
    array = parent(B)
    array .= zero(array)
    return B
end