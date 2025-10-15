module TJProdEst

    using DataFrames, LinearAlgebra, ShiftedArrays, Optim, PrettyTables
    include("AuxFncs.jl")
    include("EstimationFncs.jl")

    export tj_prod_est
end