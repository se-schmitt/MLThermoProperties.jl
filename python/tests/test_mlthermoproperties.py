import numpy as np
from pytest import approx
import pyclapeyron as cl
import mlthermoproperties as mlp


def test_hanna_grappa():
    model = mlp.HANNA(["ethanol", "benzene"], puremodel=mlp.GRAPPA)
    p, _, _, y = cl.bubble_pressure(model, 333.15, np.array([0.5, 0.5]))
    assert p == approx(74307.8, rel=1e-4)
    assert y == approx([0.4636, 0.5364], abs=1e-3)


def test_user_smiles():
    smiles = {"SMILES": ["CCO", "CCOC(C)=O"]}
    model = mlp.HANNA(["ethanol", "ethyl acetate"], puremodel=mlp.GRAPPA,
                      userlocations=smiles, pure_userlocations=smiles)
    p, _, _, _ = cl.bubble_pressure(model, 330.0, np.array([0.3, 0.7]))
    assert p == approx(57672.3, rel=1e-4)


def test_ese():
    model = mlp.ESE(["ethanol", "n-decane"])
    D = mlp.inf_diffusion_coefficient(model, 1e5, 300.0, solute="ethanol", solvent="n-decane")
    assert D == approx(1.6853e-9, rel=1e-3)
    assert isinstance(mlp.inf_diffusion_coefficient(model, 1e5, 300.0), np.ndarray)
