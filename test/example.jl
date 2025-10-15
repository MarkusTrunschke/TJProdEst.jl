using Pkg; Pkg.activate(pwd())
using TJProdEst, DataFrames, CSV


data = CSV.read("test/sim_data.csv", DataFrame)

res = tj_prod_est(data = data,
                  output = :Y,
                  flexible_input = [:M],
                  fixed_inputs = [:L, :K],
                  flexible_input_price = :Pᴹ,
                  output_price = :Pʸ,
                  ω_lom_degree = 1,
                  time = :year,
                  id = :ID,
                  options = Dict{Symbol, Any}())