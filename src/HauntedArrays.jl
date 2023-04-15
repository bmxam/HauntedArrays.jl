module HauntedArrays
using MPI
using LinearAlgebra
using MPIUtils
using Random
using ForwardDiff

include("./parallel_factory.jl")
include("./Exchanger.jl")
export update_ghosts!

include("./cache.jl")

include("./Array.jl")
export HauntedArray,
    HauntedVector,
    HauntedMatrix,
    own_to_global,
    own_to_local,
    own_to_local_rows,
    owned_values,
    get_cache,
    get_comm,
    local_to_global,
    n_local_rows,
    n_own_rows

include("./interface.jl")
include("./collective.jl")
export gather

include("./reductions.jl")
include("./broadcast.jl")
include("./algebra.jl")

include("./misc.jl")

end