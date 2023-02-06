Base.parent(A::HauntedArray) = A.array

Base.size(A::HauntedArray) = size(parent(A))
# Base.getindex(A::HauntedArray, i::Int) = getindex(parent(A), i)
Base.setindex!(A::HauntedArray, v, i::Int) = setindex!(parent(A), v, i)

function Base.getindex(A::HauntedArray, i::Int)
    # println("getting element $i of rank $(MPI.Comm_rank(get_comm(A)))")
    getindex(parent(A), i)
end

function Base.similar(A::HauntedArray, ::Type{S}) where {S}
    # Parent similar
    array = similar(parent(A), S)

    # Create array without ghosts
    ownedValues = view(array, A.oids)

    return HauntedArray(array, ownedValues, A.exchanger, A.lid2gid, A.oids, A.ghids)
end

Base.similar(A::HauntedArray{T}) where {T} = similar(A, T)

function Base.zero(A::HauntedArray)
    B = similar(A)
    array = parent(B)
    array .= zero(array)
    return B
end