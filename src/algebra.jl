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

for f in (:+, :-)
    @eval function (Base.$f)(A::HauntedArray, B::HauntedArray)
        _C = $f(parent(A), parent(B))
        C = zero(A)
        C.array .= _C
        return C
    end
end