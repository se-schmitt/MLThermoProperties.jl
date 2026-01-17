using MLPROP
using Test
using JLD2
print("HW")

#viscosity_data=Dict(
#[
#("299.16",[0.0013361898137719088;0.002967824208340376;0.001068135375089325;0.0008726333026304327]),
#("319.16",[0.0009797249190180454;0.0019675387551598127;0.000752153276628083;0.000589600913219654]),
#("330.16",[0.0008429321377365204;0.0016206205177457284;0.000627838499182122;0.0004905459801094946]),
#("335.16",[0.0007903263312683715;0.0014930974133063218;0.0005798215503485786;0.0004540693997375076]),
#("345.16",[0.0006992370589951461;0.0012803730164487768;0.0004967345770736327;0.00039321550438134505])
#]
#)


validation_data=Dict(
[
("299.16",[1.6482543537961485;0.931919139410536;2.2720926704344366;2.1565516995619975]),
("319.16",[2.398242890812778;1.499678069198921;3.4423174568853114;3.4051672004992155]),
("330.16",[2.8835049909007537;1.883458270815633;4.266043969220459;4.233824652227727]),
("335.16",[3.1220121860028516;2.0752810400719994;4.689285161702368;4.643207359176968]),
("345.16",[3.633999694982114;2.4922799695456694;5.636959093505703;5.521765504779055])
]
)

Tx = [299.16;319.16;330.16;335.16;345.16]
@load "test/Viscosityfunctions.jld2" η_fun_hexadecane η_fun_ethanol η_fun_water η_fun_dodecane
model_diolane_hexadecane=SEB("C1OCOC1","CCCCCCCCCCCCCCCC",η_fun_hexadecane)
model_Acetonitrile_ethanol=SEB("CC#N","CCO",η_fun_ethanol)
model_Carbondioxide_water=SEB("O=C=O","O",η_fun_water)
model_methylal_dodecane=SEB("COCOC","CCCCCCCCCCCC",η_fun_dodecane)


Diff_methylal_dodecane=Diffusion.(model_methylal_dodecane,p_iso,Tx)*10^9
Diff_diolane_hexadecane=Diffusion.(model_diolane_hexadecane,p_iso,Tx)*10^9
Diff_Acetonitrile_ethaol=Diffusion.(model_Acetonitrile_ethanol,p_iso,Tx)*10^9
Diff_Carbondioxide_water=Diffusion.(model_Carbondioxide_water,p_iso,Tx)*10^9


@testset "MLPROP.jl" begin
@test Diff_methylal_dodecane[1] ≈ get(validation_data,"299.16",0)[1] atol=1e-5
@test Diff_diolane_hexadecane[1] ≈ get(validation_data,"299.16",0)[2] atol=1e-5
@test Diff_Acetonitrile_ethaol[1] ≈ get(validation_data,"299.16",0)[3] atol=1e-5
@test Diff_Carbondioxide_water[1] ≈ get(validation_data,"299.16",0)[4] atol=1e-5

@test Diff_methylal_dodecane[2] ≈ get(validation_data,"319.16",0)[1] atol=1e-5
@test Diff_diolane_hexadecane[2] ≈ get(validation_data,"319.16",0)[2] atol=1e-5
@test Diff_Acetonitrile_ethaol[2] ≈ get(validation_data,"319.16",0)[3] atol=1e-5
@test Diff_Carbondioxide_water[2] ≈ get(validation_data,"319.16",0)[4] atol=1e-5

@test Diff_methylal_dodecane[3] ≈ get(validation_data,"330.16",0)[1] atol=1e-5
@test Diff_diolane_hexadecane[3] ≈ get(validation_data,"330.16",0)[2] atol=1e-5
@test Diff_Acetonitrile_ethaol[3] ≈ get(validation_data,"330.16",0)[3] atol=1e-5
@test Diff_Carbondioxide_water[3] ≈ get(validation_data,"330.16",0)[4] atol=1e-5

@test Diff_methylal_dodecane[4] ≈ get(validation_data,"335.16",0)[1] atol=1e-5
@test Diff_diolane_hexadecane[4] ≈ get(validation_data,"335.16",0)[2] atol=1e-5
@test Diff_Acetonitrile_ethaol[4] ≈ get(validation_data,"335.16",0)[3] atol=1e-5
@test Diff_Carbondioxide_water[4] ≈ get(validation_data,"335.16",0)[4] atol=1e-5

@test Diff_methylal_dodecane[5] ≈ get(validation_data,"345.16",0)[1] atol=1e-5
@test Diff_diolane_hexadecane[5] ≈ get(validation_data,"345.16",0)[2] atol=1e-5
@test Diff_Acetonitrile_ethaol[5] ≈ get(validation_data,"345.16",0)[3] atol=1e-5
@test Diff_Carbondioxide_water[5] ≈ get(validation_data,"345.16",0)[4] atol=1e-5
end