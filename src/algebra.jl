# For all algebra operation, it is assumed that the input HauntedArrays are up-to-date.
# However, on the output it depends (for now) on the operation...

function LinearAlgebra.mul!(
    C::HauntedVector,
    A::HauntedMatrix,
    B::HauntedVector,
    α::Number,
    β::Number,
)
    # Arbitrary choice : A and B are assumed "up-to-date"
    # update_ghosts!(A, B)

    # Parent mul!
    mul!(parent(C), parent(A), parent(B), α, β)

    # Arbitrary choice : we choose to update C
    update_ghosts!(C)

    # I know this is weird for an in-place method,
    # but without returning C (or just @showing it),
    # the result of `A*x` is `Nothing`...
    C
end

function LinearAlgebra.norm2(A::HauntedVector)
    # MPI.Allreduce(LinearAlgebra.norm2(owned_values(A)), +, get_comm(A))
    √(mapreduce(abs2, +, A))
end
function LinearAlgebra.dot(A::HauntedVector, B::HauntedVector)
    MPI.Allreduce(LinearAlgebra.dot(owned_values(A), owned_values(B)), +, get_comm(A))
end

for f in (:+, :-)
    @eval function (Base.$f)(A::HauntedArray, B::HauntedArray)
        _C = $f(parent(A), parent(B))
        C = zero(A)
        C.array .= _C
        return C
    end
end
