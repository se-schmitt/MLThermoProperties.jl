abstract type MLIdealModel <: IdealModel end

@concrete struct MLIdealParam{SingleParam} <: EoSParam
    Es
    reference_state::ReferenceState
    Mw::SingleParam
end

@concrete struct MLIdeal{MLIdealParam, StatefulLuxLayer} <: MLIdealModel 
    components::Vector{<:AbstractString}
    params::MLIdealParam
    smodel::StatefulLuxLayer
    scaler
    references::Vector{<:AbstractString}
end

"""
    MLIdeal <: IdealModel

    MLIdeal(components;
    userlocations = String[],
    reference_state = nothing,
    verbose = false)

## Input parameters
None

## Description
Ideal model based on data from quantum mechanical (QM) simulations from *Gond et al. (2025)*.

## Model Construction Examples
```
idealmodel = MLIdeal("water")
idealmodel = MLIdeal(["water","carbon dioxide"])
```

## References
1. Gond, D., ...
"""
MLIdeal

export MLIdeal 

CL.Rgas(::MLIdeal) = R̄32

function MLIdeal(fn, smiles, E)

    lmodel, ps, st, _, _ = load_model(fn)

    # Get parameters (Mw and smiles)
    mol = RDK.get_mol(smiles)
    descs = RDK.get_descriptors(mol)
    params = MLIdealParam(
        scale(lmodel.scaler.emb, E),
        ReferenceState(),
        SingleParam("molecular weight",[smiles],Float32[descs["amw"]]),
    )

    # Create model
    smodel = StatefulLuxLayer{true}(lmodel, ps, Lux.testmode(st))
    model = MLIdeal([smiles], params, smodel, lmodel.scaler, String[])

    return model
end

function CL.a_ideal(model::MLIdeal, V, T, z)
    ∑z = sum(z)
    V⁻¹ = 1/V
    ϱ = ∑z * V⁻¹ 

    X = [
        scale(model.scaler.ϱ,ϱ);
        scale(model.scaler.T,T);
        model.params.Mw[1].*1f-3;
        model.params.Es;;
    ]

    return ∑z*first(model.smodel(X))/Rgas(model)/T
end