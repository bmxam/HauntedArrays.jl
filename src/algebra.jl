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

    error("this implementation is wrong")

    # I know this is weird for an in-place method,
    # but without returning C (or just @showing it),
    # the result of `A*x` is `Nothing`...
    C
end

# for f in (Base.:+, Base.:-)
#     @eval function ($f)(A::HauntedArray, B::HauntedArray)
#         # no need to update A and B before operation : ghost values
#         # will be wrong but we udate them on the result.
#         _C = parent(A) + parent(B)
#         C = zero(A)
#         C.array .= _C
#         update_ghosts!(C)
#         return C
#     end
# end