"""
Auxiliary functions and result container for TJProdEst.

Define the result struct at top-level (required by Julia) and provide a
constructor function `res_struct_init()` that returns a default instance.
"""

"""
    Results

A struct to store the outcome of a computation or experiment. 

# Fields
- `point_estimates::NamedTuple`: A tuple containing the point estimates of the model.
- `std_errors::NamedTuple`: A tuple containing the standard errors of the model.
- `conf_intervals::NamedTuple`: A tuple containing the confidence intervals of the model.
"""
mutable struct Results
    point_estimates::NamedTuple
    std_errors::NamedTuple
    conf_intervals::NamedTuple
end

"""
    Setup

A struct to hold the setup/configuration for the estimation procedure.

# Fields
- `data::DataFrame`: The input data for the estimation.
- `output::Symbol`: The output (dependent) variable.
- `flexible_input::Vector{Symbol}`: A vector of flexible input variable symbols.
- `fixed_inputs::Vector{Symbol}`: A vector of fixed input variable symbols.
- `flexible_input_price::Symbol`: The price variable for the flexible input.
- `output_price::Symbol`: The price variable for the output.
- `ω_lom_degree::Int`: The degree for the ω_lom polynomial.
- `time_var::Union{Symbol, Missing}`: The time variable, if applicable.
- `firm_id::Union{Symbol, Missing}`: The firm identifier variable, if applicable.
- `options::Dict{Symbol, Any}`: A dictionary of additional options for the estimation.
"""
struct Setup
    output::Symbol
    flexible_input::Symbol
    fixed_inputs::Union{Symbol,Vector{Symbol}}
    flexible_input_price::Symbol
    all_inputs::Vector{Symbol}
    output_price::Symbol
    ω_lom_degree::Int
    time::Union{Symbol, Missing}
    id::Union{Symbol, Missing}
    prd_fnc_form::String
    SE_estimation::Bool
    options::Dict{Symbol, Any}
end

"""
    res_struct_init(Setup::Setup) -> Results

Initializes and returns a new `Results` object with the specified fields.

# Arguments
- `Setup::Setup`: The setup struct containing all necessary parameters.

# Returns
- `Results`: A newly created result structure initialized with the given fields.
"""
function res_struct_init(Setup::Setup)
    # Setup point estimates tuple
    lom_sym_vec = ["ω"]
    for j = 2:Setup.ω_lom_degree
        if j > 1
            lom_sym_vec = hcat(lom_sym_vec, "ω"*string(superscript_this!(string(j))))
        end
    end
    ω_lom_tpl = (;Pair.(Symbol.(lom_sym_vec), Ref{Union{Float64,Missing}}())...)

    prd_fnc_tpl = if Setup.prd_fnc_form == "CobbDouglas"
        (; Pair.(Setup.all_inputs, Ref{Union{Float64,Missing}}())...)
    else
        error("Unsupported production function form: "*Setup.prd_fnc_form)
    end

    # Setup conf intervals tuple
    prd_fnc_tpl = (; Pair.(Setup.all_inputs, [Vector{Union{Float64,Missing}}(undef,2) for _ in 1:length(Setup.all_inputs)])...)
    ω_lom_tpl = (; Pair.(Symbol.(lom_sym_vec), [Vector{Union{Float64,Missing}}(undef,2) for _ in 1:length(lom_sym_vec)])...)

    return Results((prd_fnc = prd_fnc_tpl, ω_lom = ω_lom_tpl), (prd_fnc = prd_fnc_tpl, ω_lom = ω_lom_tpl), (prd_fnc = prd_fnc_tpl, ω_lom = ω_lom_tpl))
end

"""
    setup_struct_init(data, output, flexible_input, fixed_inputs, flexible_input_price,
                      output_price, ω_lom_degree, time, id, prd_fnc_form, options) -> Setup

Construct a `Setup` object that consolidates all inputs and options required
by the estimator. This helper normalizes input lists (it builds `all_inputs`
by concatenating `flexible_input` and `fixed_inputs`) and packages arguments
into the `Setup` struct used by the rest of the codebase.

# Arguments
- `data::DataFrame`: the input dataset.
- `output::Symbol`: dependent variable (output) column name.
- `flexible_input::Vector{Symbol}`: vector of flexible input column names.
- `fixed_inputs::Vector{Symbol}`: vector of fixed input column names.
- `flexible_input_price::Symbol`: price variable for the flexible input.
- `output_price::Symbol`: price variable for the output.
- `ω_lom_degree::Int`: degree for the ω polynomial (order of LOM terms).
- `time::Union{Symbol, Missing}`: time variable column (or `missing`).
- `id::Union{Symbol, Missing}`: firm identifier column (or `missing`).
- `prd_fnc_form::String`: production function form (e.g. "CobbDouglas").
- `options::Dict{Symbol,Any}`: extra options passed to the estimator.

# Returns
- `Setup`: a filled `Setup` struct ready to be used by the estimation routine.

# Example
```julia
setup = setup_struct_init(df, :Y, [:M], [:K,:L], :Pᴹ, :Pʸ, 1, :year, :ID, "CobbDouglas", Dict())
```
"""
function setup_struct_init(output::Symbol,
                           flexible_input::Symbol,
                           fixed_inputs::Union{Symbol,Vector{Symbol}},
                           flexible_input_price::Symbol,
                           output_price::Symbol,
                           ω_lom_degree::Int,
                           time::Union{Symbol, Missing},
                           id::Union{Symbol, Missing},
                           prd_fnc_form::String,
                           std_err_estimation::Bool,
                           options::Dict{Symbol, Any})
    return Setup(
        output,
        flexible_input,
        fixed_inputs,
        flexible_input_price,
        vcat(flexible_input, fixed_inputs),
        output_price,
        ω_lom_degree,
        time,
        id,
        prd_fnc_form,
        std_err_estimation,
        options
    )
end

## Basic data preparation function
function jt_data_prep(data::DataFrame, Setup::Setup)
    
    # Select necessary variables from data frame
    est_data = copy(data[:, [Setup.time, Setup.id, Setup.output, Setup.flexible_input, Setup.fixed_inputs..., Setup.flexible_input_price, Setup.output_price]])

    # Build variable list correctly (vcat returns a Vector{Symbol})
    vars_to_lag::Vector{Symbol} = vcat(Setup.output, Setup.fixed_inputs..., Setup.flexible_input, Setup.output_price, Setup.flexible_input_price)
    
    panel_lag!(data = est_data, id = Setup.id, time = Setup.time, variable = vars_to_lag, force = true)

    # Build lag variable names corresponding to `vars_to_lag` (e.g. :lag_K)
    lagvars = [Symbol("lag_" * string(s)) for s in vars_to_lag]

    # Add a constant to the data
    if :constant ∉ names(est_data)
        est_data.constant = ones(nrow(est_data))
    end

    # Calculate share (use the first flexible input if a vector is provided)
    flex_sym = Setup.flexible_input isa Vector ? Setup.flexible_input[1] : Setup.flexible_input
    est_data.ln_proxy_output_frac = log.((est_data[!, flex_sym] .* est_data[!, Setup.output_price]) ./ (est_data[!, Setup.output] .* est_data[!, Setup.output_price]))
    
    # Drop missings
    dropmissing!(est_data, [[Setup.time, Setup.id, Setup.output, Setup.flexible_input, Setup.fixed_inputs..., Setup.flexible_input_price, Setup.output_price]..., lagvars...])

    # Return the data
    return est_data
end

"""
        panel_lag!(; data, id, time, variable; lag_prefix="lag_", lags=1, drop_missings=false, force=false)

In-place panel lagging helper. For each `id` (panel unit) this function
computes lagged versions of the specified `variable` columns (by default a
single lag) and adds them to `data` using names prefixed by `lag_prefix`.

This function mutates `data` (hence the trailing `!`). If you prefer a
non-mutating variant use `panel_lag` (not provided here) or pass a copy of
the data to `panel_lag!`.

# Keyword arguments
- `data::DataFrame`: the data frame to mutate (must contain `id` and `time`).
- `id::Symbol`: panel identifier column.
- `time::Symbol`: time ordering column.
- `variable::Union{Vector{Symbol},Symbol}`: column or columns to lag.
- `lag_prefix::String="lag_"`: prefix for created lag column names.
- `lags::Int=1`: the lag distance (currently only single-lag is implemented in naming and checks).
- `drop_missings::Bool=false`: if true, drop rows with missing lagged values.
- `force::Bool=false`: if true, existing columns with the target lag names are removed; otherwise an error is thrown.

# Notes
- The function sorts `data` by `id` and `time` before lagging and uses
    `ShiftedArrays.lag` applied within `groupby(data, id)`.
- The newly created lag columns are named by concatenating the prefix and the
    variable name (for example `lag_prefix * string(var)` produces `"lag_K"`).
    They are inserted into `data` (mutating it). If inner lagged containers
    are mutable and shared elsewhere, changes will be visible across those
    references; this function creates independent lag columns via the
    lagging/joining process.

# Returns
- Returns the mutated `data` (also modified in-place).
"""
function panel_lag!(;data::DataFrame, id::Symbol, time::Symbol, variable::Union{Vector{Symbol},Symbol}, lag_prefix::String = "lag_", lags::Int = 1, drop_missings::Bool = false, force::Bool = false)
    
    # Clean input
    if typeof(variable) == Symbol
        variable = [variable]
    end

    if any(in(Symbol.(names(data))), Symbol.(lag_prefix, variable))
        if force == true
            select!(data, Not(filter(in(Symbol.(names(data))), Symbol.(lag_prefix, variable))))
            # df2 = df2[!, Not(Symbol.(lag_prefix, variable))]
        else
            throw("Specified name for lag of variable already present in specified dataframe. Either set force = true, choose difference lag variable name, or rename the column.")
        end
    end

    # Sort data
    sort!(data, [id, time])

    # # Do the actual lagging
    data = lagging_that_panel!(data = data, id = id, time = time, variable = variable, lag_prefix = lag_prefix, lags = lags, drop_missings = drop_missings)

    return data
end


"""
        lagging_that_panel!(; data, id, time, variable; lag_prefix="lag_", lags=1, drop_missings=false)

Internal helper that performs the low-level lagging logic used by
`panel_lag!`. This routine:

- selects `id`, `time` and the target `variable` columns
- computes lagged versions for each variable (and for `time`) using
    `ShiftedArrays.lag` within groups defined by `id`
- filters rows where the observed time difference is not equal to the
    requested `lags` (sets those lag-values to `missing`)
- joins the lagged columns back onto the original `data` and renames them
    using the provided `lag_prefix`.

# Keyword arguments
- `data::DataFrame`: the full data frame to update (it is joined to lagged values).
- `id::Symbol`: panel identifier column.
- `time::Symbol`: time ordering column.
- `variable::Union{Symbol,Vector{Symbol}}`: variable(s) to lag.
- `lag_prefix::String`: prefix for the new lag columns.
- `lags::Int`: expected lag distance (used to validate time gaps).
- `drop_missings::Bool`: if true, drop rows with missing lagged values after the join.

# Returns
- The mutated `data` with additional lag columns; the function mutates
    in-place and also returns the modified `data`.
"""
function lagging_that_panel!(;data::DataFrame, id::Symbol, time::Symbol, variable::Union{Symbol,Vector{Symbol}}, lag_prefix::String = "lag_", lags::Int = 1, drop_missings::Bool = false)

    # Generate lagged values per id and select all but the original variable (causes problems in join). The ShiftedArrays.lag function names the lagged column itself with variable_lag
    df_lag = select(data, [id, time, variable...])
    
    lag_variables::Vector{Symbol} = []

    # Lag all variables
    for lag_var in [time, variable...]
        transform!(groupby(df_lag, id), lag_var => ShiftedArrays.lag)

        push!(lag_variables, Symbol(string(lag_var) .* "_lag"))
    end

    # Drop missings in lagged variables we just generated
    dropmissing!(df_lag, lag_variables)
    
    # Check if lag is actually only the expected lag apart
    for var in lag_variables[2:end]
        df_lag[!, var] = ifelse.(df_lag[!,time] .- lags .== df_lag[!,Symbol(string(time)*"_lag")], df_lag[!,var], missing)
    end

    select!(df_lag, [time, id, lag_variables[2:end]...]) # Drop lagged time variable from df

    # Combine lagged variable with original data and sort it.
    sort!(leftjoin!(data, df_lag, on = [id, time]), [id, time])

    # Drop missings in lagged variable we just generated if user wants to
    if drop_missings == true
        dropmissing!(data, lag_variables[2:end])
    end
    
    # Rename variable to user-specified name
    for var in variable
        rename!(data, Symbol(string(var)*"_lag") => Symbol(lag_prefix*string(var)))
    end

    # Return result
    return data
end

# Define a dictionary for superscript characters
const superscript_map = Dict(
    '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
    '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹',
    'a' => 'ᵃ', 'b' => 'ᵇ', 'c' => 'ᶜ', 'd' => 'ᵈ', 'e' => 'ᵉ',
    'f' => 'ᶠ', 'g' => 'ᵍ', 'h' => 'ʰ', 'i' => 'ⁱ', 'j' => 'ʲ',
    'k' => 'ᵏ', 'l' => 'ˡ', 'm' => 'ᵐ', 'n' => 'ⁿ', 'o' => 'ᵒ',
    'p' => 'ᵖ', 'r' => 'ʳ', 's' => 'ˢ', 't' => 'ᵗ', 'u' => 'ᵘ',
    'v' => 'ᵛ', 'w' => 'ʷ', 'x' => 'ˣ', 'y' => 'ʸ', 'z' => 'ᶻ',
    '+' => '⁺', '-' => '⁻', '=' => '⁼', '(' => '⁽', ')' => '⁾'
)
"""
Function that returns the superscript of the corresponding input character (b/c Julia does not have a simple function for that)
"""
function superscript_this!(c::String) # Need to use a string as input because I don't understand Chars in Julia. Char(5) returns a different unicode than string(5). And the superscript of Char(5) does not  work
    # Return the superscript character if it exists in the map, else return the original character
    return get(superscript_map, c[1], c[1])
end