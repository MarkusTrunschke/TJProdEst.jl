"""
    tj_prod_est(; data, output, flexible_input, fixed_inputs, flexible_input_price, output_price, ω_lom_degree=1, time, id, options=Dict())

Top-level estimation entry. This function orchestrates setup and result
initialization for the production function estimation and returns a tuple
`(results, setup)` where `results` is a `Results` object and `setup` is a
`Setup` object containing the inputs and options.

# Keyword arguments
- `data::DataFrame`: input dataset
- `output::Symbol`: dependent variable column name
- `flexible_input::Vector{Symbol}`: names of flexible input variables
- `fixed_inputs::Vector{Symbol}`: names of fixed input variables
- `flexible_input_price::Symbol`: price variable for the flexible input
- `output_price::Symbol`: price variable for the output
- `ω_lom_degree::Int=1`: degree for the ω series terms (default: 1)
- `time::Symbol`: time variable column name
- `id::Symbol`: firm identifier column name
- `options::Dict{Symbol,Any}=Dict()`: additional options passed to the estimator

# Returns
- `(results::Results, setup::Setup)`

# Examples
```julia
using DataFrames
df = DataFrame(Y = rand(100), K = rand(100), L = rand(100), M = rand(100))
results, setup = tj_prod_est(data = df,
                            output = :Y,
                            flexible_input = [:M],
                            fixed_inputs = [:K, :L],
                            flexible_input_price = :Pᴹ,
                            output_price = :Pʸ,
                            ω_lom_degree = 1,
                            time = :year,
                            id = :firm,
                            options = Dict())
```
"""
function tj_prod_est(;data::DataFrame, output::Symbol, flexible_input::Vector{Symbol}, fixed_inputs::Vector{Symbol}, flexible_input_price::Symbol, output_price::Symbol, ω_lom_degree::Int=1, time::Symbol, id::Symbol, options::Dict{Symbol, Any}=Dict{Symbol, Any}())

    prd_fnc_form = "CobbDouglas"

    # Initialize struct to hold setup parameters
    Setup = setup_struct_init(data, output, flexible_input, fixed_inputs, flexible_input_price, output_price, ω_lom_degree, time, id, prd_fnc_form, options)

    # Initialize struct to hold results and meta data
    Results = res_struct_init(Setup)



    return (Results, Setup)
end

