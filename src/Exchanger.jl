abstract type AbstractExchanger end

"""
* `lid` stands for "local index" or "local identifier", meaning "on the current partition"
* `gid` stands for "global index" or "global identifier", meaning "all ranks merged"
"""
struct MPIExchanger <: AbstractExchanger
    comm::MPI.Comm
    tobesent_part2lid::Dict{Int,Vector{Int}} # to be sent to others : part => lid
    toberecv_part2lid::Dict{Int,Vector{Int}} # to be recv from others : part => lid
end

@inline get_comm(exchanger::MPIExchanger) = exchanger.comm


function MPIExchanger(comm::MPI.Comm, lid2gid, lid2part)
    #islid = convert(Vector{Bool}, lid2part .== mypart) # without `convert`, a `BitVector` is obtained

    # Create two dicts with gid
    tobesent_part2gid, toberecv_part2gid = set_up_ghosts_comm(lid2gid, lid2part, comm)

    # lid2gid --> gid2lid
    gid2lid = Dict{Int,Int}()
    for (li, gi) in enumerate(lid2gid)
        gid2lid[gi] = li
    end

    # Convert the dicts with lid
    tobesent_part2lid = Dict{Int,Vector{Int}}()
    for (ipart, gids) in tobesent_part2gid
        tobesent_part2lid[ipart] = [gid2lid[gi] for gi in gids]
    end

    toberecv_part2lid = Dict{Int,Vector{Int}}()
    for (ipart, gids) in toberecv_part2gid
        toberecv_part2lid[ipart] = [gid2lid[gi] for gi in gids]
    end

    return MPIExchanger(comm, tobesent_part2lid, toberecv_part2lid)
end

"""
Synchronize ghost values using MPI communications.

Version without buffer
"""
function update_ghosts!(array::AbstractVector, exchanger::MPIExchanger)
    # Alias
    comm = exchanger.comm
    tobesent_part2lid = exchanger.tobesent_part2lid
    toberecv_part2lid = exchanger.toberecv_part2lid

    # Buffers
    T = eltype(array) # same type for all variables for now...
    tobesent_buffers = Dict(ipart => array[lids] for (ipart, lids) in tobesent_part2lid)
    toberecv_buffers =
        Dict(ipart => zeros(T, length(lids)) for (ipart, lids) in toberecv_part2lid)

    # Receive
    recv_reqs = MPI.Request[]
    for (ipart, buffer) in toberecv_buffers
        src = ipart - 1
        push!(recv_reqs, MPI.Irecv!(buffer, src, 0, comm))
    end

    # Send
    send_reqs = MPI.Request[]
    for (ipart, buffer) in tobesent_buffers
        dest = ipart - 1
        push!(send_reqs, MPI.Isend(buffer, dest, 0, comm))
    end

    # Wait for comms to complete
    MPI.Waitall!(vcat(recv_reqs, send_reqs))

    # Update cellvars
    for (ipart, buffer) in toberecv_buffers
        lids = toberecv_part2lid[ipart]
        array[lids] .= buffer
    end
end