# Molecule graphs and group tables for COGNAC

const COGNAC_ALLOWED_Z = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
const COGNAC_BOND_AROMATIC = 4

# node feature vector of an explicit polar hydrogen: Z, degree, shifted charge, hyb, aromatic, nH
const COGNAC_H_FEATURES = (1, 1, 4, 0, 0, 0)

# outer-shell electrons per atomic number, as RDKit's `getNouterElecs` has them
const COGNAC_NOUTER = (
    1, 2,                                                       # H, He
    1, 2, 3, 4, 5, 6, 7, 8,                                     # Li - Ne
    1, 2, 3, 4, 5, 6, 7, 8,                                     # Na - Ar
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8,     # K - Kr
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8,     # Rb - Xe
    1, 2, 3, 4,                                                 # Cs - Ce
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,                # Pr - Lu
    4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8,              # Hf - Rn
    1, 2, 3, 4,                                                 # Fr - Th
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,                # Pa - Lr
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,                # Rf - Og
)

# RDKit's default valences, for implicit hydrogens and a bracketed atom's radical count
const COGNAC_VALENCES = Dict(
    :H => (1,), :B => (3,), :C => (4,), :N => (3,), :O => (2,), :F => (1,),
    :Al => (3,), :Si => (4,), :P => (3, 5), :S => (2, 4, 6), :Cl => (1,),
    :Ge => (4,), :As => (3, 5), :Se => (2, 4, 6), :Br => (1,),
    :Sn => (2, 4), :Sb => (3, 5), :Te => (2, 4, 6), :I => (1, 3, 5, 7),
    :Pb => (2, 4), :Bi => (3, 5),
)

cognac_nouter(z) = 1 <= z <= length(COGNAC_NOUTER) ? COGNAC_NOUTER[z] : 0

_other_end(edge, i) = MolecularGraph.src(edge) == i ? MolecularGraph.dst(edge) : MolecularGraph.src(edge)

# hydrogens RDKit fills in to reach the smallest allowed valence covering the bond order
function cognac_implicit_h(sym::Symbol, bond_order::Int)
    for v in get(COGNAC_VALENCES, sym, ())
        v >= bond_order && return v - bond_order
    end
    return 0
end

# `(bracketed, hydrogens in the bracket)` per SMILES atom, which MolecularGraph does not keep
function cognac_smiles_atoms(smiles::AbstractString)
    organic = ('B', 'C', 'N', 'O', 'P', 'S', 'F', 'I', 'b', 'c', 'n', 'o', 'p', 's', '*')
    s = collect(smiles)
    atoms = Tuple{Bool,Int}[]
    i = 1
    while i <= length(s)
        if s[i] == '['
            j = something(findnext(==(']'), s, i), length(s) + 1)
            token = String(s[(i + 1):min(j - 1, length(s))])
            element = match(r"^(?:\d+)?[A-Za-z][a-z]?", token)
            rest = element === nothing ? token : token[(length(element.match) + 1):end]
            h = match(r"H(\d*)", rest)
            push!(atoms, (true, h === nothing ? 0 : (isempty(h[1]) ? 1 : parse(Int, h[1]))))
            i = j + 1
        elseif s[i] in organic
            two = i < length(s) && ((s[i] == 'C' && s[i + 1] == 'l') || (s[i] == 'B' && s[i + 1] == 'r'))
            push!(atoms, (false, 0))
            i += two ? 2 : 1
        else
            i += 1
        end
    end
    return atoms
end

# the same per MolecularGraph atom, which materialises bracket hydrogens as atoms of their own
function cognac_atom_tokens(smiles::AbstractString, syms::Vector{Symbol})
    aligned = Tuple{Bool,Int}[]
    for (bracketed, h) in cognac_smiles_atoms(smiles)
        push!(aligned, (bracketed, h))
        append!(aligned, ((true, 0) for _ in 1:h))
    end
    # a SMILES the two parsers read differently falls back to "all bracketed"
    length(aligned) == length(syms) || return [(true, 0) for _ in syms]
    return aligned
end

# σ-domains of an atom (bonds plus lone pairs), which RDKit maps onto the hybridization
function cognac_domains(z, charge, valence, radicals, degree)
    z <= 1 && return degree
    nouter = cognac_nouter(z)
    free = max(0, nouter - valence - charge)
    valence + nouter - charge < 8 || return degree + free ÷ 2
    return degree + (free - radicals) ÷ 2 + radicals
end

cognac_lone_pairs(z, charge, valence) = max(0, (cognac_nouter(z) - valence - charge) ÷ 2)

# a bond is conjugated if both atoms can take part, with a π system and a further neighbour
function cognac_conjugated(neighbors, candidate, has_pi, u, v)
    (candidate[u] && candidate[v]) || return false
    (has_pi[u] || has_pi[v]) || return false
    any(w -> w != v && candidate[w], neighbors[u]) && return true
    return any(w -> w != u && candidate[w], neighbors[v])
end

# an atom can take part if it is aromatic, has an unexpanded π bond, or a free p orbital
function cognac_candidates(syms, z, charges, valences, degrees, has_pi, aroms)
    return [aroms[i] ||
            (has_pi[i] && valences[i] <= first(get(COGNAC_VALENCES, syms[i], (0,)))) ||
            (degrees[i] <= 3 && (charges[i] != 0 ||
                (z[i] < 10 && cognac_nouter(z[i]) != 7 &&
                 cognac_lone_pairs(z[i], charges[i], valences[i]) > 0)))
            for i in eachindex(syms)]
end

function cognac_molgraph(mol, smiles::AbstractString)
    syms = MolecularGraph.atom_symbol(mol)
    z = MolecularGraph.atom_number(mol)
    _z_notallowed = (!).(in.(z, Ref(MLThermoProperties.COGNAC_ALLOWED_Z)))
    if any(_z_notallowed)
        error("Element(s) [$(join(syms[_z_notallowed],','))] not allowed in COGNAC!")
    end
    charges = MolecularGraph.atom_charge(mol)
    orders = copy(MolecularGraph.bond_order(mol))
    edgelist = collect(MolecularGraph.edges(mol))
    n = length(syms)

    neighbors = [Int[] for _ in 1:n]
    bonds_of = [Int[] for _ in 1:n]
    bond_order_sum = zeros(Int, n)
    for (b, edge) in enumerate(edgelist)
        u, v = MolecularGraph.src(edge), MolecularGraph.dst(edge)
        push!(neighbors[u], v); push!(neighbors[v], u)
        push!(bonds_of[u], b); push!(bonds_of[v], b)
        bond_order_sum[u] += orders[b]; bond_order_sum[v] += orders[b]
    end

    # H atoms RDKit keeps as graph nodes: isotopes, charged, bare or bonded to another H
    keep = [syms[i] !== :H || MolecularGraph.props(mol, i).isotope != 0 || charges[i] != 0 ||
            length(neighbors[i]) != 1 || syms[first(neighbors[i])] === :H for i in 1:n]
    nodes = findall(keep)
    node_of = zeros(Int, n)
    node_of[nodes] .= eachindex(nodes)

    # hydrogens from the bracket information the SMILES carries, plus the dropped H neighbours
    tokens = cognac_atom_tokens(smiles, syms)
    implicit = [tokens[i][1] ? 0 : cognac_implicit_h(syms[i], bond_order_sum[i]) for i in 1:n]
    n_hydrogens = [implicit[i] + count(w -> !keep[w], neighbors[i]) for i in 1:n]
    valences = bond_order_sum .+ implicit

    # RDKit's `cleanUp` charge-separates a pentavalent N and an O-only hypervalent halogen
    for i in 1:n
        charges[i] == 0 || continue
        nitro = syms[i] === :N && valences[i] == 5
        halogen = syms[i] in (:Cl, :Br, :I) && valences[i] > 1 &&
                  all(w -> syms[w] === :O || syms[w] === :H, neighbors[i])
        (nitro || halogen) || continue
        for b in bonds_of[i]
            o = _other_end(edgelist[b], i)
            (orders[b] == 2 && syms[o] === :O && charges[o] == 0) || continue
            orders[b] = 1
            charges[i] += 1; valences[i] -= 1; bond_order_sum[i] -= 1
            charges[o] = -1; valences[o] -= 1; bond_order_sum[o] -= 1
            nitro && break
        end
    end

    # an unfilled bracketed valence becomes radical electrons rather than hydrogens
    radicals = zeros(Int, n)
    for i in 1:n
        tokens[i][1] || continue
        v = get(COGNAC_VALENCES, syms[i], ())
        isempty(v) && continue
        effective = cognac_nouter(z[i]) >= 4 ? first(v) + charges[i] : first(v) - charges[i]
        radicals[i] = max(0, effective - valences[i])
    end

    # MolecularGraph misses a ring its valence model rejects, and over-marks aromatic bonds
    written = [MolecularGraph.props(mol, i).isaromatic for i in 1:n]
    aroms = MolecularGraph.is_aromatic(mol) .| written
    rings = MolecularGraph.sssr(mol)
    aromatic_rings = [r for r in rings if all(i -> aroms[i], r)]
    in_aromatic_ring(u, v) = any(r -> u in r && v in r, aromatic_rings)

    degrees = [count(w -> keep[w], neighbors[i]) + n_hydrogens[i] for i in 1:n]
    domains = [cognac_domains(z[i], charges[i], valences[i], radicals[i], degrees[i]) for i in 1:n]
    has_pi = [aroms[i] || any(b -> orders[b] >= 2, bonds_of[i]) for i in 1:n]
    candidate = cognac_candidates(syms, z, charges, valences, degrees, has_pi, aroms)
    conjugated = [cognac_conjugated(neighbors, candidate, has_pi,
                      MolecularGraph.src(edgelist[b]), MolecularGraph.dst(edgelist[b]))
                  for b in eachindex(edgelist)]

    # sp3 with a conjugated bond is sp2 in RDKit, which is where an amide N gets its code
    function hyb_code(i)
        code = 2 <= domains[i] <= 6 ? domains[i] - 1 : 0
        return code == 3 && any(b -> conjugated[b], bonds_of[i]) ? 2 : code
    end

    # atoms whose hydrogens the checkpoint treats as groups of their own
    polar = [i for i in nodes if syms[i] !== :C && n_hydrogens[i] > 0]
    x = Matrix{Int}(undef, 6, length(nodes) + sum(i -> n_hydrogens[i], polar; init=0))

    for (j, i) in enumerate(nodes)
        x[:, j] .= (
            clamp(z[i], 0, 118),
            clamp(degrees[i], 0, 7),
            clamp(charges[i], -4, 4) + 4,
            hyb_code(i),
            aroms[i],
            i in polar ? 0 : clamp(n_hydrogens[i], 0, 5),
        )
    end

    bond_ring = MolecularGraph.is_edge_in_ring(mol)
    src, tgt, e = Int[], Int[], Int[]
    for (b, edge) in enumerate(edgelist)
        i, k = MolecularGraph.src(edge), MolecularGraph.dst(edge)
        u, v = node_of[i], node_of[k]
        (u > 0 && v > 0) || continue
        order = in_aromatic_ring(i, k) ? COGNAC_BOND_AROMATIC : orders[b]
        push!(src, u, v); push!(tgt, v, u)
        push!(e, order, conjugated[b], bond_ring[b], order, conjugated[b], bond_ring[b])
    end

    # append the polar hydrogens, grouped by parent, the way RDKit's `AddHs` orders them
    j = length(nodes)
    for i in polar, _ in 1:n_hydrogens[i]
        j += 1
        x[:, j] .= COGNAC_H_FEATURES
        push!(src, node_of[i], j); push!(tgt, j, node_of[i])
        push!(e, 1, 0, 0, 1, 0, 0)
    end

    return x, src, tgt, reshape(e, 3, :)
end

# consecutive ids for the distinct keys, numbered by first appearance
function _relabel(keys::AbstractVector)
    ids = Dict{eltype(keys),Int}()
    return [get!(ids, k, length(ids) + 1) for k in keys]
end

# Weisfeiler-Leman classes of one molecule's atoms, the limit of what message passing sees
function cognac_atom_classes(x, src, tgt, e)
    n = size(x, 2)
    nbrs = [Tuple{Int,Int}[] for _ in 1:n]
    for j in eachindex(src)
        push!(nbrs[tgt[j]], (src[j], e[1, j] + 5 * e[2, j] + 10 * e[3, j]))
    end

    color = _relabel([x[:, k] for k in 1:n])
    for _ in 1:n
        next = _relabel([(color[k], sort!([(color[m], b) for (m, b) in nbrs[k]]))
                         for k in 1:n])
        next == color && break
        color = next
    end

    classes = [Int[] for _ in 1:maximum(color; init = 0)]
    for k in 1:n
        push!(classes[color[k]], k)
    end
    return classes
end

# stereochemistry is not featurised, and dropping it avoids MolecularGraph's stereo perception
cognac_strip_stereo(smiles::AbstractString) = replace(smiles, r"@+" => "", '/' => "", '\\' => "")

# all molecules in one disconnected graph, with each node's molecule index and the classes
function cognac_gnngraph(smiles::AbstractVector{<:AbstractString})
    xs, es, src, tgt, molecule_index = Matrix{Int}[], Matrix{Int}[], Int[], Int[], Int[]
    classes = Vector{Int}[]
    n_nodes = 0
    for (i, s) in enumerate(cognac_strip_stereo.(smiles))
        x, s_i, t_i, e_i = cognac_molgraph(MolecularGraph.smilestomol(s), s)
        push!(xs, x); push!(es, e_i)
        append!(src, s_i .+ n_nodes); append!(tgt, t_i .+ n_nodes)
        append!(molecule_index, fill(i, size(x, 2)))
        # classes are per molecule, so atoms of different molecules are never merged
        append!(classes, (c .+ n_nodes for c in cognac_atom_classes(x, s_i, t_i, e_i)))
        n_nodes += size(x, 2)
    end

    graph = GNNGraph(src, tgt;
        num_nodes = n_nodes,
        ndata = (; x = reduce(hcat, xs)),
        edata = (; e = reduce(hcat, es)),
    )
    return graph, molecule_index, classes
end

# group parameters as the mean over the group's atoms, as in the reference implementation
function cognac_group_params(groups, R, Q, ABC)
    W = zeros(length(groups), length(R))
    for (g, atoms) in enumerate(groups)
        W[g, atoms] .= 1 / length(atoms)
    end
    merged = map(ABC) do M
        X = W * M * W'
        X .= (X .+ X') ./ 2             # symmetric up to roundoff anyway
        X[diagind(X)] .= 0              # aₖₖ = 0, exactly as for a single atom
        X
    end
    return W * R, W * Q, merged
end

# every group is unique to this mixture: "<molecule>_<first atom><element>"
function cognac_group_names(graph::GNNGraph, molecule_index, groups)
    element(z) = MolecularGraph.ATOMTABLE[z]["Symbol"]
    return ["$(molecule_index[k])_$(k)$(element(graph.ndata.x[1, k]))" for k in first.(groups)]
end

# `νₖᵢ` counts the symmetry-equivalent atoms of group `k`, as mod. UNIFAC counts "2 × CH3"
function cognac_groups(components, group_names, molecule_index, groups, grouptype::Symbol)
    group_molecule = [molecule_index[first(g)] for g in groups]
    return GroupParam(
        [(components[i], [group_names[g] => length(groups[g])
                          for g in findall(==(i), group_molecule)])
         for i in eachindex(components)],
        grouptype, String[]
    )
end
