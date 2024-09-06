# Temporary
COUNT_WARNING_FOR_VIEW = 0


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

    # Copy the cache
    cache = copy_cache(get_cache(A))

    return HauntedArray(array, A.exchanger, A.lid2gid, A.lid2part, A.oid2lid, cache)
end

Base.similar(A::HauntedArray{T}) where {T} = similar(A, T)

# Base.similar(A::HauntedArray, dims::Dims) = error("similar(A, dims::Dims), $dims")
Base.similar(A::HauntedArray{T,N}, ::Type{S}, dims::Dims{N}) where {S,T,N} = similar(A, S)

function Base.similar(A::HauntedVector, ::Type{S}, dims::Dims{2}) where {S}
    n = length(parent(A))

    # Matrix case
    @assert dims[1] == dims[2] "Only square Matrix supported for now ($dims)"
    @assert dims[1] == n "Number of rows must match number of elts of the vector"

    # Cache -> we don't reuse the infos

    return HauntedArray(get_comm(A), A.lid2gid, A.lid2part, 2, S, typeof(get_cache(A)))
end

function Base.zero(A::HauntedArray)
    B = similar(A)
    parent(B) .= zero(parent(B))
    return B
end

function Base.view(A::HauntedArray, I::Vararg{Any,N}) where {N}
    error("`view` with I = $I not implemented yet (shape of `A` = $(size(A)))")
end

function Base.view(A::HauntedMatrix, I::Vararg{Any,2})
    # A `view` on a HauntedMatrix can lead to dead-lock, because it may not
    # be collective. If the view is not intented to be used collectively, the
    # present implementation is ok, otherwise it is wrong.
    if COUNT_WARNING_FOR_VIEW == 0
        global COUNT_WARNING_FOR_VIEW += 1
        @warn "`view` for HauntedMatrix is a hack, use it only if you know what you're doing."
    end
    return view(parent(A), I...)
end

function Base.view(A::HauntedVector, I::AbstractVector)
    array = view(parent(A), I)
    exchanger = filtered_exchanger(get_exchanger(A), I)
    lid2gid = local_to_global(A)[I]
    lid2part = local_to_part(A)[I]
    cacheType = typeof(get_cache(A))

    # Build new oid2lid
    # Rq: why not just `oid2lid = findall(part -> part == mypart, lid2part)` ??
    old_li_to_old_oi = zero(local_to_global(A))
    for (oi, li) in enumerate(own_to_local(A))
        old_li_to_old_oi[li] = oi
    end
    new_li_to_old_li = I
    new_li_to_old_oi = view(old_li_to_old_oi, new_li_to_old_li)
    oid2lid = sortperm(filter(x -> x > 0, new_li_to_old_oi))

    return HauntedArray(array, exchanger, lid2gid, lid2part, oid2lid, cacheType)
end

# Usefull in case of wrapped sparse matrix. Maybe it would more clever to specialize
# directly Base.eachindex?
function Base.fill!(A::HauntedArray, x)
    fill!(parent(A), x)
    return A
end