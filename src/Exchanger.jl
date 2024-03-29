abstract type AbstractExchanger end

"""
  - `lid` stands for "local index" or "local identifier", meaning "on the current partition"
  - `gid` stands for "global index" or "global identifier", meaning "all ranks merged"

`I` maybe be `CartesianIndex` for N > 1, or `Int` for vectors

TODO : use MPI Graph API (see https://juliaparallel.org/MPI.jl/latest/examples/09-graph_communication/)
"""
struct MPIExchanger{I} <: AbstractExchanger
    comm::MPI.Comm
    tobesent_part2lid::Dict{Int,Vector{I}} # to be sent to others : part => lid
    toberecv_part2lid::Dict{Int,Vector{I}} # to be recv from others : part => lid

    function MPIExchanger(
        comm::MPI.Comm,
        tobesent_part2lid::Dict{Int,Vector{I}},
        toberecv_part2lid::Dict{Int,Vector{I}},
    ) where {I}
        new{I}(comm, tobesent_part2lid, toberecv_part2lid)
    end
end

@inline get_comm(exchanger::MPIExchanger) = exchanger.comm
@inline get_index_type(::MPIExchanger{I}) where {I} = I

function MPIExchanger(
    comm::MPI.Comm,
    lid2gid::Array{I,N},
    lid2part::Array{Int,N},
) where {N,I}
    # Create two dicts with gid
    tobesent_part2gid, toberecv_part2gid = set_up_ghosts_comm(comm, lid2gid, lid2part)

    # lid2gid --> gid2lid
    # gid2lid = Dict{Int,Int}()
    gid2lid = Dict{I,CartesianIndex{N}}()
    for li in CartesianIndices(lid2gid)
        gi = lid2gid[li]
        gid2lid[gi] = li
    end
    # @one_at_a_time display(lid2gid)
    # @one_at_a_time display(lid2part)

    # Convert the dicts with lid
    tobesent_part2lid = Dict{Int,Vector{CartesianIndex{N}}}()
    for (ipart, gids) in tobesent_part2gid
        tobesent_part2lid[ipart] = [CartesianIndex(Tuple(gid2lid[gi])) for gi in gids] # need `CartesianIndex(Tuple)` for scalar case
    end

    toberecv_part2lid = Dict{Int,Vector{CartesianIndex{N}}}()
    for (ipart, gids) in toberecv_part2gid
        toberecv_part2lid[ipart] = [CartesianIndex(Tuple(gid2lid[gi])) for gi in gids] # need `CartesianIndex(Tuple)` for scalar case
    end

    return MPIExchanger(comm, tobesent_part2lid, toberecv_part2lid)
end

"""
    update_ghosts!(array::AbstractArray, exchanger::MPIExchanger)

Synchronize ghost values using MPI communications.

Version without buffer

TODO : use `MPI.MultiRequest` instead of MPI.Request[]
"""
function update_ghosts!(array::AbstractArray, exchanger::MPIExchanger)
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
        push!(recv_reqs, MPI.Irecv!(buffer, comm; source = src))
    end

    # Send
    send_reqs = MPI.Request[]
    for (ipart, buffer) in tobesent_buffers
        dest = ipart - 1
        push!(send_reqs, MPI.Isend(buffer, comm; dest = dest))
    end

    # Wait for comms to complete
    # @one_at_a_time println("before waitall")
    MPI.Waitall(vcat(recv_reqs, send_reqs))

    # Update cellvars
    for (ipart, buffer) in toberecv_buffers
        lids = toberecv_part2lid[ipart]
        array[lids] .= buffer
    end
end

function check(comm, lid2part, root = 0)
    _lparts = unique(lid2part)
    all_parts = MPI.Gather(_lparts, comm; root = root)

    if MPI.Comm_rank(comm) == root
        @assert length(unique(all_parts)) == MPI.Comm_size(comm) "The comm size is different from the number of declared partitions"
    end
end

"""
Build a new exchanger by merging the input exchangers

`s2m` stands for "single to multi" gives, for each exchanger,
a single-exchanger numbering to a multi-exchanger numbering
"""
function merge_exchangers(
    exchangers::NTuple{N,MPIExchanger{I}},
    s2m::NTuple{N,Vector{I}},
) where {N,I}
    _tobesent_part2lid = typeof(exchangers[1].tobesent_part2lid)[]
    _toberecv_part2lid = typeof(exchangers[1].toberecv_part2lid)[]
    for (i, exchanger) in enumerate(exchangers)
        d1 = exchanger.tobesent_part2lid
        push!(_tobesent_part2lid, Dict(zip(keys(d1), s2m[i][values(d1)])))

        d2 = exchanger.toberecv_part2lid
        push!(_toberecv_part2lid, Dict(zip(keys(d2), s2m[i][values(d2)])))
    end

    tobesent_part2lid = mergewith(vcat, _tobesent_part2lid...)
    toberecv_part2lid = mergewith(vcat, _toberecv_part2lid...)

    return MPIExchanger(get_comm(exchangers[1]), tobesent_part2lid, toberecv_part2lid)
end

"""
Build a new exchanger from an existing exchanger by keeping only some local indices
"""
function filtered_exchanger(exchanger::MPIExchanger{I}, lids) where {I}
    tobesent_part2lid = Dict{Int,Vector{I}}()
    for (part, _lids) in exchanger.tobesent_part2lid
        i = intersect(_lids, lids)
        if length(i) > 0
            tobesent_part2lid[part] = i
        end
    end

    toberecv_part2lid = Dict{Int,Vector{I}}()
    for (part, _lids) in exchanger.toberecv_part2lid
        i = intersect(_lids, lids)
        if length(i) > 0
            toberecv_part2lid[part] = i
        end
    end

    return MPIExchanger(get_comm(exchanger), tobesent_part2lid, toberecv_part2lid)
end

# This exchanger does not perform any exchange, but can help debugging
struct DummyExchanger <: AbstractExchanger
    comm::MPI.Comm
end
@inline get_comm(exchanger::DummyExchanger) = exchanger.comm

# """
# Build a "Matrix" exchanger from a "Vector" exchanger
# """
# function matrix_from_vector(exchanger::MPIExchanger{I}, n::Int, vec_lid2part) where {I}
#     comm = get_comm(exchanger)
#     mypart = MPI.Comm_rank(comm) + 1

#     # Reverse toberecv_part2lid and apply min function
#     toberecv_lid2minpart = Dict{I,Int}()
#     for (part, lids) in exchanger.toberecv_part2lid
#         for lid in lids
#             if lid in keys(toberecv_lid2minpart)
#                 toberecv_lid2minpart[lid] = minimum(toberecv_lid2minpart[lid], part)
#             else
#                 toberecv_lid2minpart[lid] = part
#             end
#         end
#     end

#     # First, we init the `lid2part` with only the current partition number
#     # i.e current partition owned all its indices
#     mat_lid2part = mypart .* ones(Int, n, n)

#     # Then we loop on rows whose index are not owned by the current partition
#     # For each row-element:
#     #   * if the column index is recv from no-one, the element belongs to the current
#     #   partition
#     #   * if the column index is recv from one or more parts, the element belongs to
#     #   smallest partition id
#     #
#     # Finally, each diagonal element belongs to the partition according to lid2part
#     for (li, part_i) in enumerate(vec_lid2part)
#         if part_i != mypart
#             for (lj, part_j) in toberecv_lid2minpart
#                 mat_lid2part[li, lj] = part_j
#             end
#         end

#         mat_lid2part[li, li] = vec_lid2part[li]
#     end

#     return mat_lid2part
# end
