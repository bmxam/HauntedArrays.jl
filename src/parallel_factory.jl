"""
DEPRECATED DOCUMENTATION


Notes: there are multiple levels of elts numering for a system of variables on a distributed mesh.
Consider a dof a variable, this variable being part of a system of variables.
- this dof has a local index on one or several cells.
  Ex: dof 3 of cell 21 and dof 5 of cell 897 for instance
- this dof has a local index for the variable on the local mesh, all cells considered.
  Ex: dof 456 of this variable on the local mesh
- this dof has a local index for the entire system on the local mesh.
  Ex: dof 456 of this variable is dof 15135 of the system on the local mesh
- this dof has a global index for the entire system, all partitions considered
- this dof also has a global index for the variable (...)



* identify elts handled by local proc and elts by another partition
* filter elts not handled by local partition
* indicate number of ghost partitions and ghost partitions id (MPI.Allgather)
* send number of elts "handled" by each ghost partition
* identify all ghost elts by `ivar`, `icell_g` and `iloc` (`iloc` being the index of the dof in the cell for this variable)
* iterate:
    send them to each concerned ghost partition
    receive them from each partition
    in the received elts, apply the "min" function on the assigned partition number
    answer with the updated elts partition
* now we have an accurate `dof2part`
* count number of elts handled by local partition
* gather elts count
* compute offset from elts count and rank
* build global DofHandler with offset, assigning random number for ghost elts
* ask for the global number of ghost elts to the other partition. To do so (no need to iterate):
    indicate number of ghost partitions and number of elts handled by each ghost partition (MPI.Allgather)
    identify all ghost elts by `ivar`, `icell_g` and `iloc` (`iloc` being the index of the dof in the cell for this variable)
    send them to each concerned ghost partition
    receive them from each partition


WARNING :
- `to be recv` designates an information that is asked by the local partition (so it's an information that will be received)
- `to be sent` designates an information that is provided by the local partition (so it's an information that will be sent)
"""

function set_up_ghosts_comm(
    comm::MPI.Comm,
    lid2gid::Array{CartesianIndex{N},N},
    lid2part::Array{Int,N},
) where {N}
    my_part = MPI.Comm_rank(comm) + 1

    # Filter to obtain only ghost-index -> part
    ghid2part = Dict{CartesianIndex{N},Int}()
    for li in eachindex(CartesianIndices(lid2part))
        ipart = lid2part[li]
        if ipart != my_part
            ghid2part[CartesianIndex(Tuple(li))] = ipart # need CartesianIndex(Tuple) for scalar case
        end
    end
    ghost_parts = unique(values(ghid2part))
    #@one_at_a_time (@show ghost_parts)

    # Identify partitions that are sending infos to local partition
    src_parts = _identify_src_partitions(ghost_parts, comm)
    #@one_at_a_time (@show src_parts)

    # Count number of elts that are sent to the local partition by each `src` partition
    # (and also send this info from local part to `dest` parts)
    tobesent_part2ndofs, toberecv_part2ndofs =
        _identify_src_ndofs(ghid2part, ghost_parts, src_parts, comm)
    #@one_at_a_time (@show tobesent_part2ndofs)
    #@one_at_a_time (@show toberecv_part2ndofs)

    # Prepare buffers
    tobesent_part2gid = Dict{Int,Vector{CartesianIndex{N}}}(
        ipart => zeros(CartesianIndex{N}, tobesent_part2ndofs[ipart]) for
        ipart in src_parts
    )
    toberecv_part2gid = Dict{Int,Vector{CartesianIndex{N}}}(
        ipart => zeros(CartesianIndex{N}, toberecv_part2ndofs[ipart]) for
        ipart in ghost_parts
    )

    # Find iglobs to send to each partition
    offsets = Dict(ipart => 1 for ipart in ghost_parts)
    for (li, ipart) in ghid2part
        offset = offsets[ipart]
        toberecv_part2gid[ipart][offset] = lid2gid[li]
        offsets[ipart] += 1
    end

    # Identify iglob of elts asked by `src` partitions
    _identify_asked_gids!(toberecv_part2gid, tobesent_part2gid, comm)
    #@one_at_a_time (@show tobesent_part2iglob)

    return tobesent_part2gid, toberecv_part2gid
end



"""
Identify which proc is sending to the local proc
"""
function _identify_src_partitions(dest_parts::Vector{Int}, comm::MPI.Comm)
    n_dest_parts_loc = length(dest_parts)
    my_part = MPI.Comm_rank(comm) + 1
    nparts = MPI.Comm_size(comm)

    # Version 1 : in two steps
    # First, send number of ghost partitions for each proc
    # Second, send ghost partitions id for each proc
    # -> avoid allocation by MPI

    # Send number of ghost partitions for each proc
    n_dest_parts_glo = MPI.Allgather(n_dest_parts_loc, comm)

    # Second, send ghost partitions id for each proc
    sendrecvbuf = zeros(Int, sum(n_dest_parts_glo))
    offset = sum(n_dest_parts_glo[1:my_part-1])
    sendrecvbuf[offset+1:offset+n_dest_parts_loc] .= dest_parts
    MPI.Allgatherv!(MPI.VBuffer(sendrecvbuf, n_dest_parts_glo), comm)

    # Filter source partition targeting local partition
    src_parts = Int[]
    sizehint!(src_parts, n_dest_parts_loc) # lucky guess
    for ipart = 1:nparts
        # Skip if local part, irrelevant
        (ipart == my_part) && continue

        # Check if `my_part` is present in the ghost parts of `ipart`
        offset = sum(n_dest_parts_glo[1:ipart-1])
        if my_part ∈ sendrecvbuf[offset+1:offset+n_dest_parts_glo[ipart]]
            push!(src_parts, ipart)
        end
    end

    # Version 2 : in one steps
    # Send, in one time, the number of ghosts partitions followed by id of ghost elts for each
    #n_dest_parts = length(dest_parts)
    #send_buffer = [n_dest_parts, dest_parts...]

    return src_parts
end

"""
Count the number of elts that each `src` partition will send to the local partition. This
necessitates in return for the local partition to send this info to `dest` partitions.
"""
function _identify_src_ndofs(
    destdof2part::Dict,
    dest_parts::Vector{Int},
    src_parts::Vector{Int},
    comm::MPI.Comm,
)
    # Get the number of elts that will be sent to local partition by each src partition
    recv_reqs = MPI.Request[]
    n_src_dofs = [[0] for _ in src_parts] # need a Vector{Vector{Int}} because MPI.Irecv! can't handle an Int but only Vector{Int}
    for (i, ipart) in enumerate(src_parts)
        src = ipart - 1
        push!(recv_reqs, MPI.Irecv!(n_src_dofs[i], src, 0, comm))
    end

    # Send the number of ghost elts to each ghost partition
    toberecv_part2nelts = Dict{Int,Int}()
    send_reqs = MPI.Request[]
    for ipart in dest_parts
        n_dest_dofs = count(==(ipart), values(destdof2part))
        toberecv_part2nelts[ipart] = n_dest_dofs
        dest = ipart - 1
        push!(send_reqs, MPI.Isend(n_dest_dofs, dest, 0, comm))
    end

    MPI.Waitall!(vcat(recv_reqs, send_reqs))
    #MPI.Waitall!(recv_reqs) # no need to wait for the send_reqs to achieve

    tobesent_part2nelts =
        Dict(ipart => n_src_dofs[i][1] for (i, ipart) in enumerate(src_parts))
    return tobesent_part2nelts, toberecv_part2nelts
end



"""
Identify local ids that are asked by `src` partitions

Return, for each `src` partition, the local ids asked by the remote partition
"""
function _identify_asked_gids!(
    toberecv_part2gids::Dict{Int,Vector{CartesianIndex{N}}},
    tobesent_part2gids::Dict{Int,Vector{CartesianIndex{N}}},
    comm::MPI.Comm,
) where {N}
    # Receive asked elts ids
    # we want to identify which dof we will be sending to `src` partitions, so the buffer is name "to be sent"
    send_reqs = MPI.Request[]
    for (ipart, buffer) in tobesent_part2gids
        src = ipart - 1
        push!(send_reqs, MPI.Irecv!(buffer, src, 0, comm))
    end

    # Send ghost elts ids
    # we want to tell the others procs what dof we will be receiving, so the buffer is named "to be recv"
    recv_reqs = MPI.Request[]
    for (ipart, buffer) in toberecv_part2gids
        dest = ipart - 1
        push!(recv_reqs, MPI.Isend(buffer, dest, 0, comm))
    end

    # Wait for all the comm to be over
    MPI.Waitall!(vcat(send_reqs, recv_reqs))
    #MPI.Waitall!(recv_reqs)

end