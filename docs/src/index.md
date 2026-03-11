````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: MLPROP
  text: State-of-the-art hybrid thermodynamic models
  image:
    src: /assets/logo_large.png
    alt: MLPROP.jl
  tagline: Hybrid machine learning models to predict various thermodynamic properties -- from phase equilibria to transport properties 
  actions:
    - theme: alt
      text: Getting started
      link: /getting_started
    - theme: alt
      text: MLPROP models
      link: /models
    - theme: alt
      text: View on GitHub
      link: https://github.com/se-schmitt/MLPROP.jl

features:
  - icon: 🤝
    title: Thermodynamics + Machine Learning
    details: Predictions of thermodynamic properties for any substance based on hybrid thermodynamic models
    link: /models

  - icon: <img width="64" height="64" src="/assets/Clapeyron_logo_without_text.svg" alt="Clapeyron"/>
    title: Build on Clapeyron.jl
    details: Use the rich thermodynamics solvers from `Clapeyron.jl`
    link: https://github.com/ClapeyronThermo/Clapeyron.jl

  - icon: 🏃
    title: Website
    details: See our website for interactive calculation of thermodynamic properties based on the MLPROP models
    link: https://ml-prop.mv.rptu.de/
---
````

## Getting Started

Install `MLPROP.jl` in Julia by

```julia-repl
julia> using Pkg; Pkg.add("MLPROP")
```

...