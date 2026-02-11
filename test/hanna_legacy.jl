using MLPROP, ChemBERTa, Clapeyron
# Test components
# Water (1) and Ethanol (2)
# (1): "O"
# (2): "CCO"
# Gamma First Comp.:  1.455121
# Gamma Second Comp: 1.248844

# DMSO (1) and Water (2)
# (1): "CS(=O)C"
# (2): "O"
# Gamma First Comp.:  0.756031
# Gamma Second Comp: 0.552130

# Aspirin (1) and Methanol (2)
# (1): "CC(=O)Oc1ccccc1C(=O)O"
# (2): "CO"
# Gamma First Comp.:  1.197029
# Gamma Second Comp: 1.551961

# Saccharin (1) and Methanol (2)
# (1): "C1=CC=C2C(=C1)C(=O)NS2(=O)=O"
# (2): "CO"
# Gamma First Comp.:  1.185503
# Gamma Second Comp: 1.335264


@testset "HANNA_legacy" begin
    # SMILES to test
    components_list = [
        ["water", "ethanol"],                                
        ["DMSO","water"],                          
        ["aspirin","methanol"],              
        ["saccharin","methanol"],      
    ]

    gammas_ref = [
        [1.455121, 1.248844],   # water + ethanol
        [0.756031, 0.552130],   # dmso + water
        [1.197029, 1.551961],   # aspirin + methanol
        [1.185503, 1.335264]    # saccharin + methanol
    ]

    # Calculating the gammas for a given SMILES-pair and compare to Python reference
    results = Vector{Vector{Float64}}()
    for system in components_list
        model = MLPROP.HANNA(system)
        gammas = activity_coefficient(model, 1e5, 300., [.5,.5])
        push!(results, gammas)
    end
    @test results ≈ gammas_ref atol=1e-5
end