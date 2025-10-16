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
function tj_prod_est(;data::DataFrame, output::Symbol, flexible_input::Symbol, fixed_inputs::Union{Symbol,Vector{Symbol}}, flexible_input_price::Symbol, output_price::Symbol, ω_lom_degree::Int=1, time::Symbol, id::Symbol, std_err_estimation::Bool = True, options::Dict{Symbol, Any}=Dict{Symbol, Any}())

    prd_fnc_form = "CobbDouglas"

    # Unify fixed_inputs to always be a Vector
    if isa(fixed_inputs, Symbol)
        fixed_inputs = [fixed_inputs]
    end

    # Initialize struct to hold setup parameters
    Setup = setup_struct_init(output, flexible_input, fixed_inputs, flexible_input_price, output_price, ω_lom_degree, time, id, prd_fnc_form, std_err_estimation, options)

    # Initialize struct to hold results and meta data
    Results = res_struct_init(Setup)

    # Set up data
    est_data = jt_data_prep(data, Setup)

    # Estimate parameters
    tj_prodest_estimation!(data = est_data, Setup = Setup, Results = Results)


    return (Results, Setup)
end

"""
        tj_prodest_estimation!(; data, Setup, Results)

Run the production estimation pipeline for the provided dataset. This
high-level helper performs the core estimation steps and populates the
`Results` object with point estimates and (optionally) standard error
information.

# Keyword arguments
- `data::DataFrame`: prepared data to use for estimation (usually the
    output of `jt_data_prep`).
- `Setup::Setup`: configuration struct describing inputs, options and
    model form.
- `Results::Results`: mutable results container that will be filled by the
    estimation routines.

# Behavior
- Calls the single-step estimator `tj_onestep_estimator` to compute point
    estimates and writes them into `Results`.
- If `Setup.std_err_estimation` is true, calls `tj_se_estimation!` to
    compute standard errors (mutates `Results`).

# Side effects
- This function mutates the `Results` object in-place. It does not return
    a new `Results` instance; it returns `nothing` implicitly.

# Example
```julia
# prepare data and setup
results, setup = tj_prod_est(data = df, output = :Y, flexible_input = [:M], fixed_inputs = [:K,:L], flexible_input_price = :Pᴹ, output_price = :Pʸ, ω_lom_degree = 1, time = :year, id = :ID)
# run lower-level estimation directly (results is mutated)
tj_prodest_estimation!(data = prepared_df, Setup = setup, Results = results)
```
"""
function tj_prodest_estimation!(;data::DataFrame,
                               Setup::Setup,
                               Results::Results)

        tj_onestep_estimator(data = data, Setup = Setup, Results = Results)

        if Setup.std_err_estimation
                tj_se_estimation!(data = data, Setup = Setup, Results = Results)
        end
end
