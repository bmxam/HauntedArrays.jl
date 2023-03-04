module HauntedArrays
using MPI
using LinearAlgebra
using MPIUtils

include("./parallel_factory.jl")
include("./Exchanger.jl")
export update_ghosts!

include("./Array.jl")
export HauntedArray, HauntedVector, owned_values, owned_indices, get_comm

# include("./Vector.jl")
# export HauntedVector, enable_ghosts, disable_ghosts

include("./interface.jl")
include("./collective.jl")
export gather

include("./reductions.jl")
include("./broadcast.jl")
include("./algebra.jl")


end