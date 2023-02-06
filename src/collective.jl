update_ghosts!(A::HauntedArray) = update_ghosts!(parent(A), get_exchanger(A))

gather(v::HauntedArray, root = 0) = MPI.gather(parent(v), root, comm)