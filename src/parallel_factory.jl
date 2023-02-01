"""
Notes: there are multiple levels of dofs numering for a system of variables on a distributed mesh.
Consider a dof a variable, this variable being part of a system of variables.
- this dof has a local index on one or several cells.
  Ex: dof 3 of cell 21 and dof 5 of cell 897 for instance
- this dof has a local index for the variable on the local mesh, all cells considered.
  Ex: dof 456 of this variable on the local mesh
- this dof has a local index for the entire system on the local mesh.
  Ex: dof 456 of this variable is dof 15135 of the system on the local mesh
- this dof has a global index for the entire system, all partitions considered
- this dof also has a global index for the variable (...)



* identify dofs handled by local proc and dofs by another partition
* filter dofs not handled by local partition
* indicate number of ghost partitions and ghost partitions id (MPI.Allgather)
* send number of dofs "handled" by each ghost partition
* identify all ghost dofs by `ivar`, `icell_g` and `iloc` (`iloc` being the index of the dof in the cell for this variable)
* iterate:
    send them to each concerned ghost partition
    receive them from each partition
    in the received dofs, apply the "min" function on the assigned partition number
    answer with the updated dofs partition
* now we have an accurate `dof2part`
* count number of dofs handled by local partition
* gather dofs count
* compute offset from dofs count and rank
* build global DofHandler with offset, assigning random number for ghost dofs
* ask for the global number of ghost dofs to the other partition. To do so (no need to iterate):
    indicate number of ghost partitions and number of dofs handled by each ghost partition (MPI.Allgather)
    identify all ghost dofs by `ivar`, `icell_g` and `iloc` (`iloc` being the index of the dof in the cell for this variable)
    send them to each concerned ghost partition
    receive them from each partition


WARNING :
- `to be recv` designates an information that is asked by the local partition (so it's an information that will be received)
- `to be sent` designates an information that is provided by the local partition (so it's an information that will be sent)
"""

function set_up_ghosts_comm(loc2glob::Vector{Int}, dof2part::Vector{Int}, comm::MPI.Comm)
    my_part = MPI.Comm_rank(comm) + 1

    # Filter to obtain only ghost-dof -> part
    ghostdof2part = Dict{Int,Int}()
    for (idof_l, ipart) in enumerate(dof2part)
        if ipart != my_part
            ghostdof2part[idof_l] = ipart
        end
    end
    ghost_parts = unique(values(ghostdof2part))
    #@one_at_a_time (@show ghost_parts)

    # Identify partitions that are sending infos to local partition
    src_parts = _identify_src_partitions(ghost_parts, comm)
    #@one_at_a_time (@show src_parts)

    # Count number of dofs that are sent to the local partition by each `src` partition
    # (and also send this info from local part to `dest` parts)
    tobesent_part2ndofs, toberecv_part2ndofs =
        _identify_src_ndofs(ghostdof2part, ghost_parts, src_parts, comm)
    #@one_at_a_time (@show tobesent_part2ndofs)
    #@one_at_a_time (@show toberecv_part2ndofs)

    # Prepare buffers
    tobesent_part2iglob = Dict{Int,Vector{Int}}(
        ipart => zeros(Int, tobesent_part2ndofs[ipart]) for ipart in src_parts
    )
    toberecv_part2iglob = Dict{Int,Vector{Int}}(
        ipart => zeros(Int, toberecv_part2ndofs[ipart]) for ipart in ghost_parts
    )

    # Find iglobs to send to each partition
    offsets = Dict(ipart => 1 for ipart in ghost_parts)
    for (idof_l, ipart) in ghostdof2part
        offset = offsets[ipart]
        toberecv_part2iglob[ipart][offset] = loc2glob[idof_l]
        offsets[ipart] += 1
    end

    # Identify iglob of dofs asked by `src` partitions
    _identify_asked_iglob!(toberecv_part2iglob, tobesent_part2iglob, comm)
    #@one_at_a_time (@show tobesent_part2iglob)

    return tobesent_part2iglob, toberecv_part2iglob
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
    # Send, in one time, the number of ghosts partitions followed by id of ghost dofs for each
    #n_dest_parts = length(dest_parts)
    #send_buffer = [n_dest_parts, dest_parts...]

    return src_parts
end

"""
Count the number of dofs that each `src` partition will send to the local partition. This
necessitates in return for the local partition to send this info to `dest` partitions.
"""
function _identify_src_ndofs(
    destdof2part::Dict{Int,Int},
    dest_parts::Vector{Int},
    src_parts::Vector{Int},
    comm::MPI.Comm,
)
    # Get the number of dofs that will be sent to local partition by each src partition
    recv_reqs = MPI.Request[]
    n_src_dofs = [[0] for _ in src_parts] # need a Vector{Vector{Int}} because MPI.Irecv! can't handle an Int but only Vector{Int}
    for (i, ipart) in enumerate(src_parts)
        src = ipart - 1
        push!(recv_reqs, MPI.Irecv!(n_src_dofs[i], src, 0, comm))
    end

    # Send the number of ghost dofs to each ghost partition
    toberecv_part2ndofs = Dict{Int,Int}()
    send_reqs = MPI.Request[]
    for ipart in dest_parts
        n_dest_dofs = count(==(ipart), values(destdof2part))
        toberecv_part2ndofs[ipart] = n_dest_dofs
        dest = ipart - 1
        push!(send_reqs, MPI.Isend(n_dest_dofs, dest, 0, comm))
    end

    MPI.Waitall!(vcat(recv_reqs, send_reqs))
    #MPI.Waitall!(recv_reqs) # no need to wait for the send_reqs to achieve

    tobesent_part2ndofs =
        Dict(ipart => n_src_dofs[i][1] for (i, ipart) in enumerate(src_parts))
    return tobesent_part2ndofs, toberecv_part2ndofs
end



"""
Identify local sys dof index that are asked by `src` partitions

Return, for each `src` partition, the local sys dof ids asked by the remote partition
"""
function _identify_asked_iglob!(
    toberecv_part2dofs::Dict{Int,Vector{Int}},
    tobesent_part2dofs::Dict{Int,Vector{Int}},
    comm::MPI.Comm,
)
    # Receive asked dofs ids
    # we want to identify which dof we will be sending to `src` partitions, so the buffer is name "to be sent"
    send_reqs = MPI.Request[]
    for (ipart, buffer) in tobesent_part2dofs
        src = ipart - 1
        push!(send_reqs, MPI.Irecv!(buffer, src, 0, comm))
    end

    # Send ghost dofs ids
    # we want to tell the others procs what dof we will be receiving, so the buffer is named "to be recv"
    recv_reqs = MPI.Request[]
    for (ipart, buffer) in toberecv_part2dofs
        dest = ipart - 1
        push!(recv_reqs, MPI.Isend(buffer, dest, 0, comm))
    end

    # Wait for all the comm to be over
    MPI.Waitall!(vcat(send_reqs, recv_reqs))
    #MPI.Waitall!(recv_reqs)

end

"""
Update `dof2part` by asking to the ghost partition who is the owner of each ghost dof.

`tobesent_idofs` are the dofs asked by remote partitions : it is an info that the local partition knows and send to others
`toberecv_idofs` are the dofs unknown to the local partition : it is an info asked by the local partition to the others
"""
function _update_dof2part!(
    dof2part::Vector{Int},
    tobesent_idofs::Dict{Int,Vector{Int}},
    toberecv_idofs::Dict{Int,Vector{Int}},
    comm::MPI.Comm,
)
    # Async recv
    recv_reqs = MPI.Request[]
    recv_dof2part = Dict{Int,Vector{Int}}(
        ipart => zeros(Int, size(toberecv_idofs[ipart])) for ipart in keys(toberecv_idofs)
    )
    for ipart in keys(toberecv_idofs)
        dest = ipart - 1
        buffer = recv_dof2part[ipart]
        push!(recv_reqs, MPI.Irecv!(buffer, dest, 0, comm))
    end

    # Async send
    send_reqs = MPI.Request[]
    for (ipart, idofs_l) in tobesent_idofs
        dest = ipart - 1
        buffer = dof2part[idofs_l] # current MPI doesn't support `view(dof2part, idofs_l)``
        push!(send_reqs, MPI.Isend(buffer, dest, 0, comm))
    end

    MPI.Waitall!(vcat(send_reqs, recv_reqs))
    #MPI.Waitall!(recv_reqs)

    # Deal with answers
    #converged = true # for an unknown reason, using Bool fails on the supercomputer
    _converged = 1 # ... so we use integers instead
    for (ipart, idofs_l) in toberecv_idofs
        _recv_dof2part = recv_dof2part[ipart]
        for i = 1:length(idofs_l)
            idof_l = idofs_l[i]
            if dof2part[idof_l] > _recv_dof2part[i]
                dof2part[idof_l] = _recv_dof2part[i]
                #converged = false
                _converged = 0
            end
        end
    end

    # Need to obtain the status of all procs
    #MPI.Allreduce!(keep_going, MPI.LOR, comm)
    #converged = MPI.Allreduce(converged, MPI.LOR, comm)
    converged = Bool(MPI.Allreduce(_converged, MPI.LOR, comm))

    return converged
end

"""
Update `loc2glob`.
`send_idofs_l` is a dict (ipart => idofs_l)
"""
function _update_loc2glob!(
    loc2glob::Vector{Int},
    tobesent_idofs_l::Dict{Int,Vector{Int}},
    toberecv_idofs_l::Dict{Int,Vector{Int}},
    comm::MPI.Comm,
)
    # Async recv
    recv_reqs = MPI.Request[]
    recv_idofs_g = Dict{Int,Vector{Int}}(
        ipart => zeros(Int, size(toberecv_idofs_l[ipart])) for
        ipart in keys(toberecv_idofs_l)
    )
    for ipart in keys(toberecv_idofs_l)
        dest = ipart - 1
        buffer = recv_idofs_g[ipart]
        push!(recv_reqs, MPI.Irecv!(buffer, dest, 0, comm))
    end

    # Async send
    send_reqs = MPI.Request[]
    for (ipart, idofs_l) in tobesent_idofs_l
        dest = ipart - 1
        buffer = loc2glob[idofs_l] # current MPI doesn't support `view(dof2part, idofs_l)`
        push!(send_reqs, MPI.Isend(buffer, dest, 0, comm))
    end

    MPI.Waitall!(vcat(send_reqs, recv_reqs))
    #MPI.Waitall!(recv_reqs)

    # Deal with answers
    #converged = true # for an unknown reason, using Bool fails on the supercomputer
    _converged = 1
    for (ipart, idofs_l) in toberecv_idofs_l
        _recv_idofs_g = recv_idofs_g[ipart]
        for i = 1:length(idofs_l)
            idof_l = idofs_l[i]
            # Check that the received dof is relevant AND this dof is not already known
            # (the second part is important for convergence)
            if ((_recv_idofs_g[i] > 0) && (loc2glob[idof_l] == 0))
                loc2glob[idof_l] = _recv_idofs_g[i]
                #converged = false
                _converged = 0
            end
        end
    end

    # Need to obtain the status of all procs
    #MPI.Allreduce!(keep_going, MPI.LOR, comm)
    #converged = MPI.Allreduce(converged, MPI.LAND, comm)
    converged = Bool(MPI.Allreduce(_converged, MPI.LAND, comm))

    return converged
end


""" debug function to print the location of the dofs (only for Lagrange 1) """
function dof2coords(sys, mesh)
    @assert get_nvars(sys) == 1 "only valid for scalar system"
    c2n = connectivities_indices(mesh, :c2n)
    printed_dofs = zeros(Bool, get_ndofs(sys))
    for icell = 1:ncells(mesh)
        ct = cells(mesh)[icell]
        n = get_nodes(mesh, c2n[icell])
        for (ivar, cv) in enumerate(get_cvs(sys))
            idofs = dof(sys, cv, icell)
            idof_by_vertex(function_space(cv), shape(ct))
            for i = 1:length(idofs)
                if !printed_dofs[idofs[i]]
                    xyz = n[i].x
                    println("$ivar, $(idofs[i]) <--> ($(xyz[1]), $(xyz[2]))")
                    printed_dofs[idofs[i]] = true
                end
            end
        end
    end
end

""" debug function """
function printcellcenters(mesh)
    c2n = connectivities_indices(mesh, :c2n)
    for icell = 1:ncells(mesh)
        n = get_nodes(mesh, c2n[icell])
        c = Bcube.center(n, cells(mesh)[icell])
        println("$icell <--> ($(c[1]), $(c[2])")
    end
end