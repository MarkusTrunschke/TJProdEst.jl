using TJProdEst
using Documenter

DocMeta.setdocmeta!(TJProdEst, :DocTestSetup, :(using TJProdEst); recursive=true)

makedocs(;
    modules=[TJProdEst],
    authors="Markus Trunschke <markus.trunschke@googlemail.com>",
    sitename="TJProdEst.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
