abstract type AbstractCache end

function build_cache(
    C::Type{AbstractCache},
    exchanger::AbstractExchanger,
    lid2gid::Vector{I},
    lid2part::Vector{Int},
    oid2lid,
    ndims::Int,
    T,
) where {I<:Integer}
    error("`build_cache` not implemented for $C")
end

copy_cache(c::AbstractCache) = error("`copy_cache` not implemented for $(typeof(c))")

struct EmptyCache <: AbstractCache end

function build_cache(
    ::Type{EmptyCache},
    ::AbstractExchanger,
    ::Vector{I},
    ::Vector{Int},
    _,
    ::Int,
    _,
) where {I}
    return EmptyCache()
end

copy_cache(::EmptyCache) = EmptyCache()