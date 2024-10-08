module HauntedArrays
using MPI
using LinearAlgebra
using MPIUtils
using Random

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
    owned_by_me,
    owned_values,
    get_cache,
    get_comm,
    local_to_global,
    local_to_part,
    n_local_rows,
    n_own_rows,
    set_owned_values

include("./interface.jl")
include("./collective.jl")
export gather

include("./reductions.jl")
include("./broadcast.jl")
include("./algebra.jl")

include("./misc.jl")

end
