using DataStructures
using SparseArrays
using Statistics
import Base.show

#### VANTAGE POINT TREE FOR NEIGHBORS

struct VPNode
    index::Int
    data::Vector{Float64}
    radius::Float64
    min_dist::Float64
    max_dist::Float64
    n::Int
    left_child::Union{VPNode,Nothing}
    right_child::Union{VPNode,Nothing}
end

Base.show(io::IO, node::VPNode) = print("VPNode with n=$(node.n), ind=$(node.index), vp=$(node.data), r=$(node.radius)")

"""
Recursive function to construct the Vantage Point Tree
"""
function build_VPTree(data::Matrix{Float64})
    ind_data = collect(enumerate([data[i,:] for i in 1:size(data)[1]]))
    _build_VPTree(ind_data)
end

function _build_VPTree(data::Vector{Tuple{Int,Vector{Float64}}})
    # Stopping criteria
    if isempty(data)
        return nothing
    end
    N = length(data)
    if N == 1
        return  VPNode(data[1][1], data[1][2], 0., 0., 0., N, nothing, nothing)
    end
    # Else, choose a VP
    i = rand(1:N)
    vp = data[i]
    # Then find a permutation of the other points that presents them in increasing distance from vp
    idx = vcat(1:i-1, i+1:N)
    other = data[idx]
    dist = [eu_dist(x[2], vp[2]) for x in other]
    i_split = (N-1) ÷ 2 + 1
    dist_perm = sortperm(dist, alg=PartialQuickSort(i_split))
    permute!(other, dist_perm)
    # Divide the points for the LEFT and RIGHT
    left_points = other[1:i_split]
    right_points = other[i_split+1:N-1]
    # Call the recursion on both
    left_node = _build_VPTree(left_points)
    right_node = _build_VPTree(right_points)
    min_d, max_d = extrema(dist)
    radius = dist[dist_perm[i_split]]

    VPNode(vp[1], vp[2], radius, min_d, max_d, N, left_node, right_node)
end

function find_neighbors(tree_root::VPNode, point::Vector{Float64}, n_neigh::Int)
    tau = Inf
    to_search = [tree_root]
    results = PriorityQueue{Tuple{Int,Vector{Float64}},Float64}(Base.Order.Reverse)
    it = 0
    while length(to_search) > 0
        curr = pop!(to_search)
        d = eu_dist(point, curr.data)
        if length(results) >= n_neigh
            frt_i, frt_d = peek(results)
            tau = eu_dist(frt_i[2], point)
        end
        if d < tau && d > eps()
            enqueue!(results, (curr.index, curr.data), d)
            if length(results) > n_neigh
                dequeue!(results)
            end
        end
        if d - curr.radius <= tau && !isnothing(curr.left_child) && d - curr.min_dist >= -tau
            pushfirst!(to_search, curr.left_child)
            it += 1
        end
        if d - curr.radius > -tau && !isnothing(curr.right_child) && d - curr.max_dist <= tau
            pushfirst!(to_search, curr.right_child)
            it += 1
        end
    end
    neighbors = []
    for i in 1:n_neigh
        push!(neighbors, dequeue!(results)[1])
    end
    neighbors
end

function get_neighbors(data::Matrix{Float64}, vp_root::VPNode, n_neigh::Int)
    n, d = size(data)
    n_neigh >= n && throw(ArgumentError("n_neigh must be < $n, chosen $n_neigh: try reducing perplexity"))  
    neighbors = Dict{Int,Vector{Int}}()
    for i in 1:size(data)[1]
        N = find_neighbors(vp_root, data[i,:], n_neigh)
        neighbors[i] = N
    end
    neighbors
end

function cond_gauss_sparseVP(data::Matrix{Float64}, sigma::Vector{Float64}, perpl::Float64)
    vp_root = build_VPTree(data)
    n_neigh = Int(floor(3 * perpl))
    N_dic = get_neighbors(data, vp_root, n_neigh)
    P = spzeros(size(data)[1], size(data)[1])
    for k in keys(N_dic)
        P[k,N_dic[k]] = cond_gauss(vcat(data[k,:]', data[N_dic[k],:]), sigma=sigma[k])[1][2:end,1]
    end
    P, N_dic
end

joint_gauss_sparse(P::SparseMatrixCSC{Float64, Int64}) = (P + P') ./ (2 * size(P)[1])

#### QUADTREES FOR BARNES-HUT

mutable struct NodeQT
    # parent::Union{Nothing,NodeQT}
    children::Dict{String,Union{Nothing,NodeQT}}
    data::Vector{Vector{Float64}}
    center::Vector{Float64}
    lims::Vector{Vector{Float64}}
    cm::Vector{Float64}
    n::Int
    NodeQT(children, data, center, lims, cm, n) = new(children, data, center, lims, cm, n)
end

function get_quadrant(center::Vector{Float64}, point::Vector{Float64})
    return point[1] ≤ center[1] ? (point[2] ≤ center[2] ? "sw" : "nw") :
                                  (point[2] > center[2] ? "ne" : "se")
end

function empty_children()
    Dict([("nw",nothing), ("ne",nothing), ("sw",nothing), ("se",nothing)])
end

Base.show(io::IO, node::NodeQT) = print("NodeQT with n=$(node.n), center=$(node.center), lims=$(node.lims)")
NodeQT(center::Vector{Float64}, lims::Vector{Vector{Float64}}) = NodeQT(empty_children(), [], center, lims, copy(center), 0)
NodeQT(center::Vector{Float64}, lims::Vector{Vector{Float64}}, cm::Vector{Float64},n::Int) = NodeQT(empty_children(), [], center, lims, cm, n)

function create_root(data::Matrix{Float64})
    lims = [vcat(minimum(data, dims=1), maximum(data, dims=1))[:,i] for i in 1:size(data)[2]]
    center = [mean(l) for l in lims]
    cm = mean(data, dims=1)[1,:]
    n = size(data)[1]
    NodeQT(center, lims, cm, n)
end

function get_center_limits(node::NodeQT, Q::String)
    center = [mean([node.lims[i][1], node.center[i]]) for i in 1:length(node.center)]
    lims = [[node.lims[i][1], node.center[i]] for i in 1:length(node.lims)]
    if Q == "nw"
        lims[2] = [node.center[2], node.lims[2][2]]
        center[2] = mean(lims[2])
    elseif Q == "ne"
        x_lim = [node.center[1], node.lims[1][2]]
        y_lim = [node.center[2], node.lims[2][2]]
        center = [mean(x_lim), mean(y_lim)]
        lims = [x_lim, y_lim]
    elseif Q == "se"
        lims[1] = [node.center[1], node.lims[1][2]]
        center[1] = mean(lims[1])
    end
    return center, lims
end

function update_node!(node::NodeQT, point::Vector{Float64})
    node.cm = [node.cm[i] * node.n + point[i] for i in 1:length(point)] / (node.n + 1)
    node.n += 1
    node.data = push!(node.data,point)
end

global t = 0

function build_QTree(root::Union{Nothing,NodeQT}, data_::Union{Vector{Vector{Float64}},Matrix{Float64}})
    isnothing(root) && (root = create_root(data_))
    N = typeof(data_) == Matrix{Float64} ? size(data_)[1] : length(data_)
    global t += 1
    # Loop on every point
    for i in 1:N
        if t > 1000
            throw(OverflowError("There's a problem with the recursion"))
        end
        p = typeof(data_) == Matrix{Float64} ? data_[i,:] : data_[i]
        Q = get_quadrant(root.center, p)
        if isnothing(root.children[Q])
            center, lims = get_center_limits(root, Q)
            root.children[Q] = NodeQT(center, lims)
            update_node!(root.children[Q], p)
        else
            update_node!(root.children[Q], p)
            if root.children[Q].n == 2
                points = root.children[Q].data
            else
                points = [p]
            end
            root.children[Q] = build_QTree(root.children[Q], points)
        end
    end
    return root
end

F_rep_Z(a::Vector{Float64},b::Vector{Float64}) = (-1. /(1. +eu_dist(a, b)^2))^2 .* (a - b)

Z(a::Vector{Float64},b::Vector{Float64}) = (1. /(1. +eu_dist(a, b)^2))

function compute_forces(node::NodeQT, point::Vector{Float64}, theta::Float64)
    grad_rep = zeros(length(point))
    z = 0.
    s = abs(node.lims[1][2] - node.lims[1][1]) * sqrt(2) # diagonal
    d = eu_dist(point, node.cm)
    if node.n == 1
        if point ∉ node.data
            grad_rep += F_rep_Z(point, node.data[1])
            z += Z(point, node.data[1])
        end
    elseif s/d < theta
        grad_rep += node.n * F_rep_Z(point, node.cm)
        z += node.n * Z(point, node.cm)
    else
        for k in keys(node.children)
            if !isnothing(node.children[k])
                g_, z_ = compute_forces(node.children[k], point, theta)
                grad_rep += g_
                z += z_
            end
        end
    end
    return grad_rep, z
end


function gradient_trees(data_red::Matrix{Float64}, P_joint::SparseMatrixCSC{Float64, Int64}, neighbors::Dict{Int,Vector{Int}}; theta::Float64=0.2)
    N, d = size(data_red)
    # F_attr
    Q_Z = spzeros(N, N)
    F_attr = zeros(N, d)
    n_neigh = length(neighbors[1])
    for k in keys(neighbors)
        for j in neighbors[k]
            Q_Z[k,j] = Z(data_red[k,:], data_red[j,:])
        end
        Y_kj = repeat(data_red[k,:]', n_neigh,1) - data_red[neighbors[k],:]
        F_attr[k,:] = sum(Array(P_joint[k, neighbors[k]]) .* Array(Q_Z[k, neighbors[k]]) .* Y_kj, dims=1)
    end
    # F_rep
    qtree = build_QTree(nothing, data_red)
    F_rep = zeros(N, d)
    for i in 1:N
        g, z = compute_forces(qtree, data_red[i,:], theta)
        F_rep[i,:] = g / z
    end
    4. * (F_attr + F_rep)
end

    