abstract type AbstractExchanger end

"""
* `lid` stands for "local index" or "local identifier", meaning "on the current partition"
* `gid` stands for "global index" or "global identifier", meaning "all ranks merged"

`I` maybe be `CartesianIndex` for N > 1, or `Int` for vectors
"""
struct MPIExchanger{I} <: AbstractExchanger where {I}
    comm::MPI.Comm
    tobesent_part2lid::Dict{Int,Vector{I}} # to be sent to others : part => lid
    toberecv_part2lid::Dict{Int,Vector{I}} # to be recv from others : part => lid
end

@inline get_comm(exchanger::MPIExchanger) = exchanger.comm

function MPIExchanger(
    comm::MPI.Comm,
    lid2gid::Array{I,N},
    lid2part::Array{Int,N},
) where {N,I}
    # Create two dicts with gid
    tobesent_part2gid, toberecv_part2gid = set_up_ghosts_comm(comm, lid2gid, lid2part)

    # lid2gid --> gid2lid
    # gid2lid = Dict{Int,Int}()
    gid2lid = Dict()
    for (li, gi) in enumerate(lid2gid)
        gid2lid[gi] = li
    end

    # Convert the dicts with lid
    tobesent_part2lid = Dict{Int,Vector{CartesianIndex{N}}}()
    for (ipart, gids) in tobesent_part2gid
        tobesent_part2lid[ipart] = [CartesianIndex(Tuple(gid2lid[gi])) for gi in gids] # need `CartesianIndex(Tuple)` for scalar case
    end

    toberecv_part2lid = Dict{Int,Vector{CartesianIndex{N}}}()
    for (ipart, gids) in toberecv_part2gid
        toberecv_part2lid[ipart] = [CartesianIndex(Tuple(gid2lid[gi])) for gi in gids] # need `CartesianIndex(Tuple)` for scalar case
    end

    return MPIExchanger{I}(comm, tobesent_part2lid, toberecv_part2lid)
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