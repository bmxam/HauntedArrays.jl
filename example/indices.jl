module test
A = [10 72 43; 54 25 16]
B = ones(4)

rows = [1 3 2]
cols = [3 2 1]

# lid2gid est un array
# lid2gid[1, 1] -> renvoie un CartesianIndex

#c = CartesianIndices(rows, cols)
l = LinearIndices(A)
c = CartesianIndices(A)

for _c in eachindex(c)
    @show _c, typeof(_c)
end

e = CartesianIndex(1, 2)
f = CartesianIndex(e...)

# for _c in eachindex(A)
#     @show _c, typeof(_c)
# end

# for _c in eachindex(CartesianIndices(B))
#     @show _c, typeof(_c)
# end


end