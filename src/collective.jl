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
function gather(v::HauntedArray, root = 0)
    error("to be updated with new design")

    comm = get_comm(v)

    nloc = length(v.oids)
    n_by_rank = MPI.Allgather(nloc, comm)
    nmax = maximum(n_by_rank)

    _lid2gid = similar(v.lid2gid, nmax)
    _lid2gid[1:nloc] = v.lid2gid[v.oids]
    _lid2gid = MPI.Gather(_lid2gid, root, comm)

    _values = similar(v.array, nmax)
    _values[1:nloc] = v.array[v.oids]
    _values = MPI.Gather(_values, root, comm)

    if MPI.Comm_rank(comm) == root
        ntot = sum(n_by_rank)
        lid2gid = similar(_lid2gid, ntot)
        values = similar(_values, ntot)

        iabs = 1
        for ip = 1:MPI.Comm_size(comm)
            for i = 1:n_by_rank[ip]
                lid2gid[iabs] = _lid2gid[(ip-1)*nmax+i]
                values[iabs] = _values[(ip-1)*nmax+i]
                iabs += 1
            end
        end

        # Reshape and reorder
        ndims = length(lid2gid[1])
        dims = ntuple(d -> maximum(cartIndex -> cartIndex[1], lid2gid), ndims)
        array = similar(values, dims)
        for (cartIndex, value) in zip(lid2gid, values)
            array[cartIndex] = value
        end

        return array
    end
end