# mlthermoproperties

Python interface to [MLThermoProperties.jl](https://github.com/se-schmitt/MLThermoProperties.jl), a collection of hybrid ML models for thermodynamic property prediction (HANNA, GRAPPA, COGNAC, ESE, HADES).
The models are [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) models and work with all functions of [pyclapeyron](https://pypi.org/project/pyclapeyron/).

## Installation

- `uv`

```bash
uv add mlthermoproperties
```

- `pip`

```bash
pip install mlthermoproperties
```

On the first import, Julia and all required Julia packages are installed and precompiled automatically (this takes several minutes once).

## Example

```python
import numpy as np
import pyclapeyron as cl
import mlthermoproperties as mlp

model = mlp.HANNA(["ethanol", "benzene"], puremodel=mlp.GRAPPA)
p, v_l, v_v, y = cl.bubble_pressure(model, 333.15, np.array([0.5, 0.5]))
```

Diffusion coefficients with ESE:

```python
model = mlp.ESE(["ethanol", "n-decane"])
D = mlp.inf_diffusion_coefficient(model, 1e5, 300.0, solute="ethanol", solvent="n-decane")
```

Compositions have to be passed as NumPy arrays.
The Julia packages are updated with:

```bash
uv run python -c "import juliapkg; juliapkg.update()"
```

See the [documentation](https://se-schmitt.github.io/MLThermoProperties.jl/stable) for details on the models.
