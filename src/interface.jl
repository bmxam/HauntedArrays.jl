Base.parent(A::HauntedArray) = A.array

Base.size(A::HauntedArray) = size(parent(A))
Base.setindex!(A::HauntedArray, v, i::Int) = setindex!(parent(A), v, i)
function Base.setindex!(A::HauntedArray, v, I::Vararg{Int,N}) where {N}
    setindex!(parent(A), v, I...)
end

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

    return HauntedArray(array, A.exchanger, A.lid2gid, A.lid2part, A.oid2lid)
end

Base.similar(A::HauntedArray{T}) where {T} = similar(A, T)

# Base.similar(A::HauntedArray, dims::Dims) = error("similar(A, dims::Dims), $dims")
Base.similar(A::HauntedArray{T,N}, ::Type{S}, dims::Dims{N}) where {S,T,N} = similar(A, S)

function Base.similar(A::HauntedVector, ::Type{S}, dims::Dims{2}) where {S}
    n = length(parent(A))

    # Matrix case
    @assert dims[1] == dims[2] "Only square Matrix supported for now ($dims)"
    @assert dims[1] == n "Number of rows must match number of elts of the vector"

    return HauntedArray(get_comm(A), A.lid2gid, A.lid2part, 2, S)
end

function Base.zero(A::HauntedArray)
    B = similar(A)
    parent(B) .= zero(parent(B))
    return B
end