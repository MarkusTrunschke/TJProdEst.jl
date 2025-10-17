module TJProdEst

    using DataFrames, LinearAlgebra, ShiftedArrays, Optim, PrettyTables, Statistics, Random, Distributions, ProgressMeter
    include("AuxFncs.jl")
    include("EstimationFncs.jl")

    export tj_prod_est
end