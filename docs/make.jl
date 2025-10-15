using TJProdEst
using Documenter

DocMeta.setdocmeta!(TJProdEst, :DocTestSetup, :(using TJProdEst); recursive=true)

makedocs(;
    modules=[TJProdEst],
    authors="Markus Trunschke <markus.trunschke@googlemail.com>",
    sitename="TJProdEst.jl",
    # Explicitly set the repository URL so Documenter doesn't need a valid git remote
    repo = "https://github.com/MarkusTrunschke/TJProdEst.jl",
    format=Documenter.HTML(;
        edit_link="main",
        # Link to repository root used for navbar/source links
        repolink = "https://github.com/MarkusTrunschke/TJProdEst.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
