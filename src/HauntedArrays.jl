module HauntedArrays
using MPI

include("./utils.jl")
export @one_at_a_time, @only_root, @only_proc

include("./parallel_factory.jl")
include("./Exchanger.jl")
export update_ghosts!

include("./Array.jl")
export HauntedArray, HauntedVector, owned_values, get_comm

# include("./Vector.jl")
# export HauntedVector, enable_ghosts, disable_ghosts

include("./interface.jl")
include("./collective.jl")
include("./reductions.jl")
include("./broadcast.jl")


end