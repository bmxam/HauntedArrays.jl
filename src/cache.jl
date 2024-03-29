abstract type AbstractCache end

function build_cache(
    C::Type{AbstractCache},
    array::AbstractArray,
    exchanger::AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oid2lid;
    kwargs...,
) where {I<:Integer}
    error("`build_cache` not implemented for $C")
end

copy_cache(c::AbstractCache) = error("`copy_cache` not implemented for $(typeof(c))")

struct EmptyCache <: AbstractCache end

function build_cache(
    ::Type{EmptyCache},
    ::AbstractArray,
    ::AbstractExchanger,
    ::Vector{I},
    ::Vector{Int},
    ::Vector{I};
    kwargs...,
) where {I}
    return EmptyCache()
end

copy_cache(::EmptyCache) = EmptyCache()
