function LinearAlgebra.mul!(
    C::HauntedArray,
    A::HauntedArray,
    B::HauntedArray,
    α::Number,
    β::Number,
)
    update_ghosts!(A, B)
    mul!(parent(C), parent(A), parent(B), α, β)
    update_ghosts!(C)

    # I know this is weird for an in-place method,
    # but without returning C (or just @showing it),
    # the result of `A*x` is `Nothing`...
    C
end