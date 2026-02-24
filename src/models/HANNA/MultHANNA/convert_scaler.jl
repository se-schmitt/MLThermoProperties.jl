using PythonCall
using JLD2

script_dir = @__DIR__
project_path = normpath(joinpath(script_dir, "..", "..", "..", ".."))
scaler_dir = joinpath(project_path, "test", "data", "python_multHANNA", "utils", "scalers")

out_dir = joinpath(project_path, "database", "multHANNA")

# load python moduls
pickle = pyimport("pickle")
builtins = pyimport("builtins")

# scaler_T
f_T = builtins.open(joinpath(scaler_dir, "temperature_scaler.pkl"), "rb")
py_scaler_T = pickle.load(f_T)
f_T.close()

mean_T = pyconvert(Array, py_scaler_T.mean_)
scale_T = pyconvert(Array, py_scaler_T.scale_)

# bert_scaler
f_emb = builtins.open(joinpath(scaler_dir, "bert_scaler.pkl"), "rb")
py_scaler_emb = pickle.load(f_emb)
f_emb.close()

mean_emb = pyconvert(Array, py_scaler_emb.mean_)
scale_emb = pyconvert(Array, py_scaler_emb.scale_)

# save as .jld2
jldsave(joinpath(out_dir, "scaler_T_multhanna.jld2"); μ=mean_T[1], σ=scale_T[1])
jldsave(joinpath(out_dir, "scaler_emb_multhanna.jld2"); μ=mean_emb, σ=scale_emb)

println("Scaler converting complete.")
