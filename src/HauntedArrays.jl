module HauntedArrays
using MPI
using LinearAlgebra
using MPIUtils

include("./parallel_factory.jl")
include("./Exchanger.jl")
export update_ghosts!

include("./Array.jl")
export HauntedArray,
    HauntedVector, own_to_local_rows, owned_values, get_comm, local_to_global

include("./interface.jl")
include("./collective.jl")
export gather

include("./reductions.jl")
include("./broadcast.jl")
include("./algebra.jl")


end