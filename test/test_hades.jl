@testitem "HADES" begin 
    using EntropyScaling, PythonCall

    p_iso = 1e5 

    models_bin = Dict(
        # [smiles] => ([D_self_ref], [smiles_ref], [pure_vis_ref])
        ["acetone", "cyclohexane"]      => ([2.944136625831489, 2.4975122610231852], ["CC(C)=O", "C1CCCCC1"], [0.000307, 0.0008932]),
        ["DMSO", "water"]               => ([0.3080188063146804, 0.701327616454973], ["CS(=O)C", "O"], [0.00199, 0.000891]),
    )        
    # Compare to https://ml-prop.mv.rptu.de
    for (system_i, (D_self_ref_i, smiles_i, vis_i)) in models_bin
        model_bin = HADES(
            system_i; 
            userlocations=(;SMILES=smiles_i),
            vismodel=[ConstantModel(Viscosity(), vis_i[1]),ConstantModel(Viscosity(), vis_i[2])]   # Viscosity in Pa*s
        )    
        D_self_bin = self_diffusion_coefficient(model_bin, p_iso, 298.15, [0.5, 0.5]) .*1e9
        @test all(isapprox.(D_self_bin, D_self_ref_i; rtol=1e-5))
    end

    # ternary
    model_ter = HADES(["ethanol", "water", "methanol"]; 
            userlocations=(;SMILES=["CCO","O","CO"]),
            vismodel=[ConstantModel(Viscosity(), 0.00104),
                    ConstantModel(Viscosity(), 0.000891),
                    ConstantModel(Viscosity(), 0.000543)]   # Viscosity in Pa*s
        )    

    D_self_ter_ref = [1.0479798013974382, 1.670611759640624, 1.267051753143833]
    D_self_ter = self_diffusion_coefficient(model_ter, p_iso, 298.15, [0.2, 0.4, 0.4]) .*1e9
    @test all(isapprox.(D_self_ter, D_self_ter_ref; rtol=1e-5))
end