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
    data::DataFrame
    output::Symbol
    flexible_input::Vector{Symbol}
    fixed_inputs::Vector{Symbol}
    flexible_input_price::Symbol
    all_inputs::Vector{Symbol}
    output_price::Symbol
    ω_lom_degree::Int
    time::Union{Symbol, Missing}
    id::Union{Symbol, Missing}
    prd_fnc_form::String
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
function setup_struct_init(data::DataFrame,
                           output::Symbol,
                           flexible_input::Vector{Symbol},
                           fixed_inputs::Vector{Symbol},
                           flexible_input_price::Symbol,
                           output_price::Symbol,
                           ω_lom_degree::Int,
                           time::Union{Symbol, Missing},
                           id::Union{Symbol, Missing},
                           prd_fnc_form::String,
                           options::Dict{Symbol, Any})
    return Setup(
        data,
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
        options
    )
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