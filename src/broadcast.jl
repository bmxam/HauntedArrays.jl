Base.BroadcastStyle(::Type{<:HauntedVector}) = Broadcast.ArrayStyle{HauntedVector}()

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{HauntedVector}},
    ::Type{ElType},
) where {ElType}
    # Scan the inputs for the HauntedVector:
    A = find_ha(bc)

    return similar(A, ElType) # not sure this is enough
end

"`A = find_ha(As)` returns the first HauntedVector among the arguments."
find_ha(bc::Base.Broadcast.Broadcasted) = find_ha(bc.args)
find_ha(args::Tuple) = find_ha(find_ha(args[1]), Base.tail(args))
find_ha(x) = x
find_ha(::Tuple{}) = nothing
find_ha(a::HauntedVector, rest) = a
find_ha(::Any, rest) = find_ha(rest)