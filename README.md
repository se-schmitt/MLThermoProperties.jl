[![Dev][docs-stable-img]][docs-stable-url] [![Dev][docs-dev-img]][docs-dev-url] [![Build Status][build-img]][build-url] # [![Paper][paper-img]][paper-url]

<p>
  <picture>
    <!-- <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logos/logo_dark.svg"> -->
    <!-- <source media="(prefers-color-scheme: light)" srcset="docs/src/assets/logos/logo.svg"> -->
    <img src="docs/src/assets/logos/logo_with_text_left.svg">
  </picture>
</p>

<!-- # MLThermoProperties.jl -->

This repository contains Julia implementations of hybrid ML models for thermodynamic property prediction integrated with [Clapeyron.jl](https://github.com/ClapeyronThermo/Clapeyron.jl) as the thermodynamic solver library.

The documentation for `MLThermoProperties.jl` can be found [here](https://se-schmitt.github.io/MLThermoProperties.jl/stable).

An interactive website for MLPROP is available at [https://ml-prop.mv.rptu.de](https://ml-prop.mv.rptu.de).

## Examples

- **VLE calculation with HANNA and GRAPPA**: compute the bubble point of the equimolar mixture ethanol + benzene at 333.15 K, using HANNA for the activity coefficients and GRAPPA for the pure-component vapor pressures:

  ```julia
  julia> using MLThermoProperties, Clapeyron

  julia> model = ogHANNA(["ethanol", "benzene"]; puremodel = GRAPPA)
  ogHANNA with 2 components:
   "ethanol"
   "benzene"
  Contains parameters: emb, scaler_T, nn, Mw

  julia> p, _, _, y = bubble_pressure(model, 333.15, [0.5, 0.5]);

  julia> p, y
  (75464.8842312711, [0.456908488637454, 0.543091511362546])
  ```

- **Activity coefficients with HANNA**: predict activity coefficients for ethanol + benzene at 333.15 K with the original HANNA model:

  ```julia
  julia> using MLThermoProperties, Clapeyron

  julia> model = ogHANNA(["ethanol", "benzene"]; puremodel = PR)
  ogHANNA with 2 components:
   "ethanol"
   "benzene"
  Contains parameters: emb, scaler_T, nn, Mw

  julia> activity_coefficient(model, 1e5, 333.15, [0.5, 0.5])
  2-element Vector{Float64}:
   1.431696397770973
   1.6120775659687212
  ```

- **Infinite-dilution diffusion coefficients with ESE**: compute the infinite-dilution diffusion coefficient of ethanol in n-decane at 300 K:

  ```julia
  julia> using MLThermoProperties, EntropyScaling, CoolProp

  julia> model = ESE(["ethanol", "n-decane"])
  ESE with 2 components:
   "ethanol"
   "n-decane"
  Contains parameters: b_ij, Mw

  julia> inf_diffusion_coefficient(model, 1e5, 300.0; solute = "ethanol", solvent = "n-decane")
  1.68527894058434e-9
  ```

More complete workflows, including p-x-y diagrams and temperature sweeps, are available in the [documentation](https://se-schmitt.github.io/MLThermoProperties.jl/stable).

## ChemBERTa.jl

[ChemBERTa.jl](lib/ChemBERTa) is a small, independently usable subpackage that provides a customized [ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR) encoder for generating molecular embeddings from SMILES strings. It is used internally by the MLPROP models and can also be used directly:

```julia
using ChemBERTa

model = ChemBERTa.load()
embedding = model("CCCO")
```

See the [ChemBERTa.jl README](lib/ChemBERTa/README.md) for details.

## Citation

If you use `MLThermoProperties.jl`, please cite the GitHub repository:

```bibtex
@misc{MLThermoProperties.jl,
  author = {Sebastian Schmitt and contributors},
  title = {MLThermoProperties.jl},
  howpublished = {\url{https://github.com/se-schmitt/MLThermoProperties.jl}}
}
```

An accompanying paper is in preparation and will be added here when available.

## Contributing

Bug reports, questions, feature requests, and improvements are welcome. Please open an [issue](https://github.com/se-schmitt/MLThermoProperties.jl/issues) to start a discussion or submit a [pull request](https://github.com/se-schmitt/MLThermoProperties.jl/pulls) with a proposed change.

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://se-schmitt.github.io/MLThermoProperties.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://se-schmitt.github.io/MLThermoProperties.jl/dev

[build-img]: https://github.com/se-schmitt/MLThermoProperties.jl/actions/workflows/CI.yml/badge.svg?branch=main
[build-url]: https://github.com/se-schmitt/MLThermoProperties.jl/actions/workflows/CI.yml?query=branch%3Amain

[paper-img]: https://img.shields.io/badge/paper-general-blue.svg
[paper-url]: https://doi.org/10.1002/cite.70004