# Using MLThermoProperties from Python

This tutorial shows how to use `MLThermoProperties.jl` from Python.
The connection is made via [`pyclapeyron`](https://pypi.org/project/pyclapeyron/), the Python interface of [`Clapeyron.jl`](https://github.com/ClapeyronThermo/Clapeyron.jl), which runs Julia in the background through [`juliacall`](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).
All models of `MLThermoProperties.jl` are regular `Clapeyron.jl` models, so every property and phase-equilibrium function of `pyclapeyron` works with them directly.

## Setup

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment.
Create a new project and add `pyclapeyron`, `rdkit`, and `matplotlib` (for plotting):

```bash
uv init mlprop-python
cd mlprop-python
uv add pyclapeyron rdkit matplotlib
```

`pyclapeyron` only installs `Clapeyron.jl` on the Julia side.
`MLThermoProperties.jl` is added to the Julia environment of the project with [`juliapkg`](https://github.com/JuliaPy/pyjuliapkg), which is installed together with `juliacall`:

```bash
uv run python -c "import juliapkg; juliapkg.add('MLThermoProperties', '5b10a24f-5c48-4ecd-8f3d-34970e95074b')"
```

The dependency is stored inside `.venv`, so this command has to be repeated if the virtual environment is recreated.
On the first `import juliacall`, Julia is installed (if no compatible Julia is found), and all Julia packages are installed and precompiled.
This takes several minutes but is only done once.

!!! note "Why rdkit?"
    When Julia is started from Python, `ChemBERTa.jl` uses the Python `rdkit` package to canonicalize SMILES.
    Without `rdkit`, an error is printed when loading `MLThermoProperties` and models built from user-defined SMILES fail.

## Building a model

Julia packages and functions are accessed through `juliacall.Main`.
We load `MLThermoProperties` and build a [`HANNA`](@ref MLThermoProperties.multHANNA) model for ethanol + benzene, using [`GRAPPA`](@ref MLThermoProperties.GRAPPA) for the pure-component vapor pressures:

```python
import numpy as np
import pyclapeyron as cl
from juliacall import Main as jl

jl.seval("using MLThermoProperties")

model = jl.HANNA(["ethanol", "benzene"], puremodel=jl.GRAPPA)
jl.display(model)
```

```
multHANNA with 2 components:
 "ethanol"
 "benzene"
Contains parameters: emb, θs, scaler_T, nn, Mw
```

The first model construction in a Python session takes 30 to 60 s because Julia compiles the neural networks.
Later calls in the same session are fast.

Models from `Clapeyron.jl` are available directly from `pyclapeyron`, e.g., `puremodel=cl.PR` uses the Peng-Robinson EOS for the pure components.

## Calculating properties

The functions of `pyclapeyron` take the `MLThermoProperties` model like any other `Clapeyron` model.
Compositions have to be passed as NumPy arrays (`pyclapeyron` rejects Python lists of numbers), and results are returned as NumPy arrays.

```python
T = 333.15
x = np.array([0.5, 0.5])

γ = cl.activity_coefficient(model, 1e5, T, x)
p, v_l, v_v, y = cl.bubble_pressure(model, T, x)
```

```
γ = [1.431 1.568]
p = 74307.8 Pa,  y = [0.464 0.536]
```

## p-x-y diagram

The bubble pressure is calculated over the whole composition range and plotted with `matplotlib`:

```python
import matplotlib.pyplot as plt

x2s = np.linspace(0, 1, 51)
p_bub, y2s = np.zeros_like(x2s), np.zeros_like(x2s)
for i, x2 in enumerate(x2s):
    p_bub[i], _, _, y = cl.bubble_pressure(model, T, np.array([1 - x2, x2]))
    y2s[i] = y[1]

fig, ax = plt.subplots()
ax.plot(x2s, p_bub / 1e3, "r-", label="bubble curve")
ax.plot(y2s, p_bub / 1e3, "b-", label="dew curve")
ax.set(xlabel="x₂, y₂ / mol mol⁻¹", ylabel="p / kPa", xlim=(0, 1))
ax.legend()
plt.show()
```

The azeotrope is found at ``x_2 \approx 0.55`` and ``p \approx 74.4`` kPa.

## User-defined SMILES

Components that are not in the `Clapeyron.jl` database can be specified by their SMILES via `userlocations`.
Python dictionaries are converted automatically.
As `GRAPPA` is built separately, the SMILES also have to be passed to it via `pure_userlocations`:

```python
smiles = {"SMILES": ["CCO", "CCOC(C)=O"]}
model = jl.HANNA(["ethanol", "ethyl acetate"], puremodel=jl.GRAPPA,
                 userlocations=smiles, pure_userlocations=smiles)
```

## Other models

All other models are used in the same way, e.g., [`GRAPPA`](@ref MLThermoProperties.GRAPPA) for vapor pressures or [`COGNAC`](@ref MLThermoProperties.COGNACexp) for activity coefficients:

```python
grappa = jl.GRAPPA(["toluene"])
psat, _, _ = cl.saturation_pressure(grappa, 383.15)

cognac = jl.COGNAC(["ethanol", "benzene"], puremodel=cl.PR)
p, _, _, y = cl.bubble_pressure(cognac, 333.15, np.array([0.5, 0.5]))
```

The diffusion models [`ESE`](@ref MLThermoProperties.ESE) and [`HADES`](@ref MLThermoProperties.HADES) build on `EntropyScaling.jl`, which has no Python wrapper.
`EntropyScaling` and `CoolProp` are added with `juliapkg` as above:

```bash
uv run python -c "import juliapkg; juliapkg.add('EntropyScaling', '36a13073-9e6e-4b19-b12b-cb530e79ee8e'); juliapkg.add('CoolProp', 'e084ae63-2819-5025-826e-f8e611a84251')"
```

Their functions are called through `jl`.
Julia arrays are returned as `juliacall.ArrayValue` and can be converted with `np.asarray`:

```python
jl.seval("using EntropyScaling, CoolProp")

ese = jl.ESE(["ethanol", "n-decane"])
D = jl.inf_diffusion_coefficient(ese, 1e5, 300.0, solute="ethanol", solvent="n-decane")
D_matrix = np.asarray(jl.inf_diffusion_coefficient(ese, 1e5, 300.0))
```
