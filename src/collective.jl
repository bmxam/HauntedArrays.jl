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

function _cartesian_oid2lid(A::HauntedArray)
    # I don't understand why, but the below array is a matrix and not a vector
    x = [CartesianIndex(i) for i in Iterators.product(_own_to_local_ndims(A)...)]
    return vec(x)
end

function _cartesian_oid2gid(A::HauntedArray)
    # I don't understand why, but the below array is a matrix and not a vector
    x = [CartesianIndex(i) for i in Iterators.product(_own_to_global_ndims(A)...)]
    return vec(x)
end

"""
Return a vector of the owned values of the Array, according to the `_cartesian_oids`
ordering.
"""
_cartesian_owned_values(A::HauntedArray) = view(parent(A), _cartesian_oid2lid(A))

"""
Gather the HauntedArray on the root process. Return an `Array`

# Warning : wrong for SparseArrays
"""
function gather(A::HauntedArray, root = 0)
    comm = get_comm(A)

    (MPI.Comm_size(comm) == 1) && return parent(A)

    # Count number of element sent by each proc, and maximum of these
    cartOwnedValues = _cartesian_owned_values(A)
    nloc = length(cartOwnedValues)
    n_by_rank = MPI.Allgather(nloc, comm)
    nmax = maximum(n_by_rank)

    # Gather the values on the root partition (each part sends the same
    # number of elts)
    _values = similar(cartOwnedValues, nmax)
    _values[1:nloc] .= cartOwnedValues
    _values = MPI.Gather(_values, root, comm)

    lid2gid = gather_lid2gid(A, root)

    if MPI.Comm_rank(comm) == root
        ntot = sum(n_by_rank)
        values = similar(_values, ntot)

        i_val = 1
        for ip = 1:MPI.Comm_size(comm)
            for i = 1:n_by_rank[ip]
                values[i_val] = _values[(ip - 1) * nmax + i]
                i_val += 1
            end
        end

        # Reshape and reorder
        dims = ntuple(d -> maximum(cartIndex -> cartIndex[d], lid2gid), ndims(A))
        array = zeros(eltype(values), dims)
        for (gid, value) in zip(lid2gid, values)
            array[gid] = value
        end

        return array
    end
end

function gather_lid2gid(A::HauntedArray, root = 0)
    comm = get_comm(A)

    (MPI.Comm_size(comm) == 1) && return A.lid2gid[own_to_local_rows(A)]

    # Count number of element sent by each proc, and maximum of these
    cartOwn2Glo = _cartesian_oid2gid(A)
    nloc = length(cartOwn2Glo)
    n_by_rank = MPI.Allgather(nloc, comm)
    nmax = maximum(n_by_rank)

    # Gather the lid2gid on the root partition (each part sends the same
    # number of elts)
    _lid2gid = similar(cartOwn2Glo, nmax)
    _lid2gid[1:nloc] .= cartOwn2Glo
    _lid2gid = MPI.Gather(_lid2gid, root, comm)

    if MPI.Comm_rank(comm) == root
        ntot = sum(n_by_rank)
        lid2gid = similar(_lid2gid, ntot)

        i_l2g = 1
        for ip = 1:MPI.Comm_size(comm)
            for i = 1:n_by_rank[ip]
                lid2gid[i_l2g] = _lid2gid[(ip - 1) * nmax + i]
                i_l2g += 1
            end
        end

        return lid2gid
    end
end