"""Utility functions for HANNA model operations."""

import os
from typing import Optional

import numpy as np
import torch
import matplotlib.tri as mtri
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from models.HANNA.HANNA import HANNA_Ensemble_Multicomponent

# Constants
DEFAULT_NUM_MODELS = 10
DEFAULT_NODES = 96
DEFAULT_EMBEDDING_SIZE = 384

def plot_predictions_binary(smiles_list, molar_fractions_all, ln_gammas, gE, hE):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].plot(molar_fractions_all[:, 0], ln_gammas[:, 0], label=smiles_list[0])
    axs[0].plot(molar_fractions_all[:, 0], ln_gammas[:, 1], label=smiles_list[1])
    axs[0].set_xlabel('$x_1$')
    axs[0].set_ylabel('$\ln \gamma_i$')
    axs[0].set_xlim(0, 1)
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('Activity Coefficients')

    axs[1].plot(molar_fractions_all[:, 0], gE)
    axs[1].set_xlabel('$x_1$')
    axs[1].set_ylabel('$g^\mathrm{E} \\, / \\, RT$')
    axs[1].set_xlim(0, 1)
    axs[1].grid(True)
    axs[1].set_title('Excess Gibbs Free Energy')

    axs[2].plot(molar_fractions_all[:, 0], hE)
    axs[2].set_xlabel('$x_1$')
    axs[2].set_ylabel('$h^\mathrm{E} \\, / \\, \\text{kJ/mol}$')
    axs[2].set_xlim(0, 1)
    axs[2].grid(True)
    axs[2].set_title('Excess Enthalpy')

    plt.show()


def _ternary_to_cartesian(molar_fractions_all):
    x1 = molar_fractions_all[:, 0]
    x2 = molar_fractions_all[:, 1]
    x3 = molar_fractions_all[:, 2]

    x = x2 + 0.5 * x3
    y = (np.sqrt(3) / 2) * x3
    return x, y, x1, x2, x3


def _style_ternary_axis(ax, smiles_list):
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.sqrt(3) / 2)
    ax.axis('off')

    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
        [0.0, 0.0],
    ])
    ax.plot(triangle[:, 0], triangle[:, 1], color='black', linewidth=1.2)

    ax.text(0.01, -0.042, smiles_list[0], ha='left', va='top', fontsize=9)
    ax.text(0.99, -0.042, smiles_list[1], ha='right', va='top', fontsize=9)
    ax.text(0.5, np.sqrt(3) / 2 + 0.010, smiles_list[2], ha='center', va='bottom', fontsize=9)


def _plot_ternary_heatmap(ax, x, y, values, smiles_list, title, levels=20):
    triangulation = mtri.Triangulation(x, y)
    contour = ax.tricontourf(triangulation, values, levels=levels, cmap='viridis')
    _style_ternary_axis(ax, smiles_list)
    ax.set_title(title, pad=24)
    return contour


def plot_predictions_ternary(smiles_list, molar_fractions_all, ln_gammas, gE, hE):
    if len(smiles_list) != 3:
        raise ValueError('plot_predictions_ternary requires exactly 3 components.')

    if molar_fractions_all.shape[1] != 3:
        raise ValueError('molar_fractions_all must have shape [N, 3] for ternary plotting.')

    x, y, _, _, _ = _ternary_to_cartesian(molar_fractions_all)
    ln_gamma_values = np.asarray(ln_gammas)

    if ln_gamma_values.shape[1] != 3:
        raise ValueError('ln_gammas must have shape [N, 3] for ternary plotting.')

    gE_values = np.asarray(gE).reshape(-1)
    hE_values = np.asarray(hE).reshape(-1)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 6)

    axs_top = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]
    ax_gE = fig.add_subplot(gs[1, 1:3])
    ax_hE = fig.add_subplot(gs[1, 3:5])

    for component_index in range(3):
        contour = _plot_ternary_heatmap(
            axs_top[component_index],
            x,
            y,
            ln_gamma_values[:, component_index],
            smiles_list,
            title=f'Activity coefficient {smiles_list[component_index]}',
            levels=20,
        )
        ln_colorbar = fig.colorbar(contour, ax=axs_top[component_index], fraction=0.046, pad=0.04)
        ln_colorbar.set_label('$\\ln \\gamma_i$')
        ln_colorbar.update_ticks()
        rounded_ticks = np.unique(np.round(ln_colorbar.get_ticks(), 2))
        ln_colorbar.set_ticks(rounded_ticks)
        ln_colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f'{value:.2f}'.rstrip('0').rstrip('.'))
        )

    gE_contour = _plot_ternary_heatmap(
        ax_gE,
        x,
        y,
        gE_values,
        smiles_list,
        title='Excess Gibbs free energy',
    )
    gE_colorbar = fig.colorbar(gE_contour, ax=ax_gE, fraction=0.046, pad=0.04)
    gE_colorbar.set_label('$g^\\mathrm{E} \\, / \\, RT$')

    hE_contour = _plot_ternary_heatmap(
        ax_hE,
        x,
        y,
        hE_values,
        smiles_list,
        title='Excess enthalpy',
    )
    hE_colorbar = fig.colorbar(hE_contour, ax=ax_hE, fraction=0.046, pad=0.04)
    hE_colorbar.set_label('$h^\\mathrm{E} \\, / \\, \\text{kJ/mol}$')

    fig.subplots_adjust(wspace=0.58, hspace=0.5)

    plt.show()

def load_ensemble(
    ensemble_path: str, 
    device: Optional[torch.device] = None, 
    num_models: int = DEFAULT_NUM_MODELS
) -> HANNA_Ensemble_Multicomponent:
    """Load HANNA ensemble model from checkpoint files.
    
    Args:
        ensemble_path: Path to directory containing ensemble model files.
        device: Device to load models on. If None, uses CPU.
        num_models: Number of models in the ensemble.
        
    Returns:
        Loaded ensemble model ready for inference.
        
    Raises:
        FileNotFoundError: If ensemble path doesn't exist.
        ValueError: If no model files found in ensemble path.
    """
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble path '{ensemble_path}' does not exist")
    
    if device is None:
        device = torch.device('cpu')
    
    # Generate model file paths
    model_paths = [
        os.path.join(ensemble_path, f'HANNA_parameters_binary{i}.pt') 
        for i in range(num_models)
    ]
    
    # Verify all model files exist
    missing_files = [path for path in model_paths if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing ensemble model files: {missing_files}"
        )
    
    print(f'Loading ensemble with {num_models} models...')
    
    ensemble = HANNA_Ensemble_Multicomponent(
        model_paths=model_paths,
        Embedding_ChemBERT=DEFAULT_EMBEDDING_SIZE,
        nodes=DEFAULT_NODES,
        device=device,
    )
    ensemble.to(device)
    
    return ensemble