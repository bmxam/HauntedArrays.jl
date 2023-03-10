"""
    update_ghosts!(A::HauntedArray)
    update_ghosts!(A::Vararg{HauntedArray,N}) where {N}

Update ghost values of the HauntedArray (involve collective communications)
"""
function update_ghosts!(::HauntedArray)
    error("`update_ghosts!` only available for `HauntedVector` for now")
end
update_ghosts!(A::HauntedVector) = update_ghosts!(parent(A), get_exchanger(A))

update_ghosts!(A::Vararg{HauntedArray,N}) where {N} = map(update_ghosts!, A) # to be improved

"""
Gather the HauntedArray on the root process. Return an `Array`
"""
function gather(A::HauntedArray{T,N}, root = 0) where {T,N}
    comm = get_comm(A)

    (MPI.Comm_size(comm) == 1) && return parent(A)

    nloc = length(owned_indices(A))^N
    n_by_rank = MPI.Allgather(nloc, comm)
    nmax_l2g = maximum(n_by_rank)
    nmax_val = nmax_l2g

    _values = similar(parent(A), nmax_val)
    _values[1:nloc] = owned_values(A)
    _values = MPI.Gather(_values, root, comm)

    lid2gid = gather_lid2gid(A, root)

    if MPI.Comm_rank(comm) == root
        ntot = sum(n_by_rank)
        values = similar(_values, ntot)

        i_val = 1
        for ip = 1:MPI.Comm_size(comm)
            for i = 1:n_by_rank[ip]
                values[i_val] = _values[(ip-1)*nmax_val+i]
                i_val += 1
            end
        end

        # Reshape and reorder
        # The line below allows for non-square matrix. If we have the square-matrix
        # hypothesis, we can compute the 'maximum' only once. But since this is a debug
        # function, no need to optimize.
        dims = ntuple(d -> maximum(cartIndex -> cartIndex[d], lid2gid), N)
        array = zeros(eltype(values), dims)
        for (gid, value) in zip(lid2gid, values)
            array[gid] = value
        end

        return array
    end
end

function gather_lid2gid(A::HauntedArray{T,N}, root = 0) where {T,N}
    comm = get_comm(A)

    (MPI.Comm_size(comm) == 1) && return A.lid2gid

    nloc = length(owned_indices(A))^N
    n_by_rank = MPI.Allgather(nloc, comm)
    nmax = maximum(n_by_rank)

    _l2g = _scalar_index_to_array_index(A.lid2gid[owned_indices(A)], N)
    _lid2gid = similar(_l2g, nmax)
    _lid2gid[1:nloc] = _l2g
    _lid2gid = MPI.Gather(_lid2gid, root, comm)

    if MPI.Comm_rank(comm) == root
        ntot = sum(n_by_rank)
        lid2gid = similar(_lid2gid, ntot)

        i_l2g = 1
        for ip = 1:MPI.Comm_size(comm)
            for i = 1:n_by_rank[ip]
                lid2gid[i_l2g] = _lid2gid[(ip-1)*nmax+i]
                i_l2g += 1
            end
        end

        return lid2gid
    end
end