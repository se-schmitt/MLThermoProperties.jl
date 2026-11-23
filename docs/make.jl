using Documenter
using DocumenterVitepress
using DocumenterCitations
using Literate
using MLThermoProperties, ChemBERTa

# for examples 
using Clapeyron, EntropyScaling, CoolProp, PythonCall 

## Tutorials -- order here defines the order shown in the sidebar
tutorials = [
    "p-x Diagram with HANNA"          => "px_diagram",
    "Diffusion coefficients with ESE" => "diffusion_coefficients",
]

## Generate tutorial markdown from Literate sources
for (_, slug) in tutorials
    Literate.markdown(
        joinpath(@__DIR__, "src", "tutorials", "$(slug).jl"),
        joinpath(@__DIR__, "src", "tutorials");
        documenter = true,
    )
end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
    sitename = "MLThermoProperties.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/se-schmitt/MLThermoProperties.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Tutorials" => [
            [title => "tutorials/$(slug).md" for (title, slug) in tutorials];
            "Using from Python" => "python.md"
        ],
        "Models" => "models.md",
        "References" => "references.md",
    ],
    plugins=[bib]
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/se-schmitt/MLThermoProperties.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)