abstract type AbstractCache{N} end
@inline get_ndims(::AbstractCache{N}) where {N} = N

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

Base.similar(c::AbstractCache) = error("`similar` not implemented for $(typeof(c))")
function Base.similar(c::AbstractCache, ::Type{S}, ::Dims{N}) where {S,N}
    error("`similar` not implemented for $(typeof(c))")
end
Base.similar(c::AbstractCache{N}, ::Type{S}, dims::Dims{N}) where {S,N} = similar(c, S)

struct EmptyCache{N} <: AbstractCache{N} end

function build_cache(
    ::Type{EmptyCache{N}},
    ::AbstractExchanger,
    ::Vector{I},
    ::Vector{Int},
    _,
    ::Int,
    _,
) where {N,I}
    return EmptyCache{N}()
end

Base.similar(::EmptyCache{N}) where {N} = EmptyCache{N}()
Base.similar(::EmptyCache{1}, ::Type{S}, dims::Dims{2}) where {S} = EmptyCache{2}()