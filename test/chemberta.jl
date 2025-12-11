@testset "ChemBERTa" begin

    

    ## Model
    

    token = encode(tokenizer, smiles)
    emb_jl_a = bert(token)
    amb_jl_b = bert(smiles)

    emb_py = readdlm("data/output.csv")

    @test 
end