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
    variance::NamedTuple
    p_values::NamedTuple
    t_statistics::NamedTuple
    conf_intervals::NamedTuple
    criterion_value::Float64
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
    ω_shifter::Union{Symbol, Vector{Symbol}}
    time::Union{Symbol, Missing}
    id::Union{Symbol, Missing}
    prd_fnc_form::String
    std_err_estimation::Bool
    std_err_type::String
    boot_reps::Int
    maximum_boot_tries::Int
    optimizer_options::NamedTuple
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
    lom_sym_vec = vcat(lom_sym_vec, string.(Setup.ω_shifter)...)
    ω_lom_tpl = (;Pair.(Symbol.(lom_sym_vec), missing)...)

    prd_fnc_tpl = if Setup.prd_fnc_form == "CobbDouglas"
        (; Pair.(vcat(:constant, Setup.all_inputs), missing)...)
    else
        error("Unsupported production function form: "*Setup.prd_fnc_form)
    end

    # Setup conf intervals tuple
    prd_fnc_ci_tpl = (; Pair.(vcat(:constant, Setup.all_inputs), [Vector{Union{Float64,Missing}}(undef,2) for _ in 1:length(Setup.all_inputs)+1])...)
    ω_lom_ci_tpl = (; Pair.(Symbol.(lom_sym_vec), [Vector{Union{Float64,Missing}}(undef,2) for _ in 1:length(lom_sym_vec)])...)

    return Results((prd_fnc = prd_fnc_tpl, ω_lom = ω_lom_tpl), # Point estimates
                   (prd_fnc = prd_fnc_tpl, ω_lom = ω_lom_tpl), # Standard errors
                   (prd_fnc = prd_fnc_tpl, ω_lom = ω_lom_tpl), # Variance
                   (prd_fnc = prd_fnc_tpl, ω_lom = ω_lom_tpl), # P-values
                   (prd_fnc = prd_fnc_tpl, ω_lom = ω_lom_tpl), # T-statistics
                   (prd_fnc = prd_fnc_ci_tpl, ω_lom = ω_lom_ci_tpl), # Confidence intervals
                   Inf) # Criterion value
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
                           ω_shifter::Union{Symbol, Vector{Symbol}},
                           time::Union{Symbol, Missing},
                           id::Union{Symbol, Missing},
                           prd_fnc_form::String,
                           std_err_estimation::Bool,
                           std_err_type::String,
                           boot_reps::Int,
                           maximum_boot_tries::Int,
                           optimizer_options::NamedTuple)

    # Default optimizer options
    default_opts = (
        lower_bound = fill(-Inf, length(fixed_inputs) + 2),
        upper_bound = fill(Inf, length(fixed_inputs) + 2),
        startvals = zeros(length(fixed_inputs) + 2) .+ 0.5,
        optimizer = Optim.NelderMead(),
        optim_options = Optim.Options()
    )

    # Merge user-provided options with defaults
    merged_opts = merge(default_opts, optimizer_options)
    
    return Setup(
        output,
        flexible_input,
        fixed_inputs,
        flexible_input_price,
        vcat(fixed_inputs, flexible_input),
        output_price,
        ω_lom_degree,
        ω_shifter,
        time,
        id,
        prd_fnc_form,
        std_err_estimation,
        std_err_type,
        boot_reps,
        maximum_boot_tries,
        merged_opts
    )
end

## Basic data preparation function
function jt_data_prep(data::DataFrame, Setup::Setup)
    
    # Select necessary variables from data frame
    est_data = copy(data[:, [Setup.time, Setup.id, Setup.output, Setup.flexible_input, Setup.fixed_inputs..., Setup.flexible_input_price, Setup.output_price, Setup.ω_shifter...]])

    # Calculate share (use the first flexible input if a vector is provided)
    est_data.ln_proxy_output_frac = log.((est_data[!, Setup.flexible_input] .* est_data[!, Setup.flexible_input_price]) ./ (est_data[!, Setup.output] .* est_data[!, Setup.output_price]))

    # Build variable list correctly (vcat returns a Vector{Symbol})
    vars_to_lag::Vector{Symbol} = vcat(Setup.output, Setup.fixed_inputs..., Setup.flexible_input, Setup.output_price, Setup.flexible_input_price, :ln_proxy_output_frac)
    
    panel_lag!(data = est_data, id = Setup.id, time = Setup.time, variable = vars_to_lag, force = true)

    # Build lag variable names corresponding to `vars_to_lag` (e.g. :lag_K)
    lagvars = [Symbol("lag_" * string(s)) for s in vars_to_lag]

    # Add a constant to the data
    if :constant ∉ names(est_data)
        est_data.constant = ones(nrow(est_data))
    end
    
    # Drop missings
    dropmissing!(est_data, [[Setup.time, Setup.id, Setup.output, Setup.flexible_input, Setup.fixed_inputs..., Setup.flexible_input_price, Setup.output_price]..., Setup.ω_shifter..., lagvars...])

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

"""
    polynomial_fnc_fast!(poly_mat, degree; par_cal=false) -> poly_mat

In-place computation of polynomial terms from a base variable. This function
mutates `poly_mat` by filling its columns with successive powers of the first
column: the i-th column is set to (first column)^i.

This is a performance-optimized version designed for repeated evaluations where
the matrix has already been preallocated. It avoids dynamic allocations at
runtime by mutating the input matrix directly (hence the trailing `!`).

# Arguments
- `poly_mat::Union{Array{<:Number}, SubArray{<:Number}}`: A matrix where the
  first column contains the base values and columns 2 through `degree` will be
  filled with polynomial terms. Must have at least `degree` columns.
- `degree::Int`: The highest polynomial degree to compute. For example, if
  `degree=3`, columns 2 and 3 will contain the squared and cubed values of
  column 1, respectively.

# Keyword Arguments
- `par_cal::Bool=false`: If `true`, uses parallel computation via `Threads.@threads`
  to compute polynomial columns concurrently.

# Returns
- Returns the mutated `poly_mat` with polynomial columns filled in-place.

# Notes
- The function assumes `poly_mat` has been preallocated with sufficient columns
  (at least `degree` columns). No bounds checking is performed.
- Column 1 is never modified; it serves as the base for all polynomial terms.
- Columns 2 through `degree` are overwritten with powers 2 through `degree`.
- For small datasets or low degrees, `par_cal=false` (sequential) is typically
  faster due to threading overhead.

# Example
```julia
# Preallocate a matrix with 3 columns for base values and polynomials up to degree 3
poly_mat = zeros(100, 3)
poly_mat[:, 1] .= rand(100)  # Fill first column with base values

# Compute polynomial terms in-place
polynomial_fnc_fast!(poly_mat, 3)
# Now poly_mat[:, 2] contains squared values, poly_mat[:, 3] contains cubed values
```
"""
function polynomial_fnc_fast!(poly_mat::Union{Array{<:Number}, SubArray{<:Number}}, degree::Int; par_cal::Bool = false)
    # Compute polynomial columns (each column of the matrix represents the i's polynomial of the first column)
    if par_cal == false
        for i in 2:degree
            poly_mat[:,i] .= @view(poly_mat[:,1]) .^ i
        end
    else
        Threads.@threads for i in 2:degree
            poly_mat[:,i] .= @view(poly_mat[:,1]) .^ i
        end
    end

    return poly_mat 
end

"""
    fastOLS(; Y, X, multicolcheck=true, force=false) -> Vector{Float64}

Compute Ordinary Least Squares (OLS) regression coefficients using optimized
linear algebra operations. This function minimizes memory allocations and uses
efficient in-place operations for computing β̂ = (X'X)⁻¹X'Y.

The implementation uses Cholesky decomposition for solving the normal equations,
which is faster than direct matrix inversion but may be less stable for
ill-conditioned design matrices.

# Keyword Arguments
- `Y::Union{Matrix, Vector}`: The dependent variable(s). Can be a vector for
  single-response regression or a matrix for multiple responses.
- `X::Union{Matrix{<:Number}, Vector{<:Number}}`: The design matrix of regressors
  (independent variables). Each column represents a regressor, each row an
  observation. Can also be a vector for single-regressor case.
- `multicolcheck::Bool=true`: If `true`, checks for perfect multicollinearity
  in the regressors by comparing `rank(X)` to the number of columns. This helps
  detect singular design matrices before attempting to solve the system.
- `force::Bool=false`: If `true` and multicollinearity is detected, automatically
  drops the first offending column and prints a warning. If `false`, throws an
  error when multicollinearity is detected. Only relevant when `multicolcheck=true`.

# Returns
- `Vector{Float64}`: A vector of estimated regression coefficients with length
  equal to the number of columns in `X` (or 1 if `X` is a vector). If `force=true`
  and a column was dropped due to multicollinearity, the length reflects the
  reduced design matrix.

# Throws
- Throws an error with message "ERROR: Regressors are perfectly multicolliniar"
  if `multicolcheck=true`, `force=false`, and `rank(X) < ncol(X)`.

# Performance Notes
- Uses in-place operations (`mul!`, `ldiv!`, `cholesky!`) to minimize allocations.
- Preallocates `cache` matrix for X'X and `coefs` vector for results.
- Cholesky decomposition via `ldiv!(cholesky!(Hermitian(cache)), coefs)` is the
  fastest approach for well-conditioned matrices (see Julia discourse #103111).
- For ill-conditioned or numerically unstable problems, consider alternative
  solvers (e.g., QR decomposition or regularized regression).

# Example
```julia
# Simple linear regression
X = [ones(100) randn(100)]  # Intercept + one regressor
Y = 2.0 .+ 3.0 .* X[:, 2] .+ 0.1 .* randn(100)
β = fastOLS(Y=Y, X=X)
# β ≈ [2.0, 3.0]

# Multiple regression with multicollinearity check
X_bad = [ones(50) randn(50) ones(50)]  # First and third columns identical
Y = randn(50)
β = fastOLS(Y=Y, X=X_bad, force=true)  # Drops one column, prints warning
```
"""
function fastOLS(; Y::Union{Matrix, Vector}, X::Union{Matrix{<:Number}, Vector{<:Number}}, multicolcheck::Bool = true, force::Bool = false) # Y::Vector{<:Number}, X::Matrix{<:Number}

    X_len::Int = 1
    if typeof(X) <: Matrix
        X_len = size(X)[2]
    end

    # Check for perfect multicoliniarity in regressors
    if multicolcheck == true && rank(X) != X_len
        if force == false
            throw("ERROR: Regressors are perfectly multicolliniar")
        else
            for j = eachindex(eachcol(X))
                if rank(X[:,Not(j)]) == X_len - 1
                    X = copy(X)
                    X = X[:,Not(j)]

                    X_len = size(X)[2]

                    println("Dropped regressor "*string(j)*" because of multicolliniarity!")
                    break
                end
            end
        end
    end

    coefs = Array{Float64}(undef, X_len, 1)
    cache = Array{Float64}(undef, X_len, X_len)
    
    # OLS
    mul!(cache, X', X) # X'X
    mul!(coefs, X', Y) # X'*Y
    ldiv!(cholesky!(Hermitian(cache)), coefs) # inv(X'*X)*X*y, see https://discourse.julialang.org/t/memory-allocation-left-division-operator/103111/3 for an explination why this is the fastest way (even though not optimal for ill-conditioned matrices)
    
    return coefs
end

## Function drawing a sample of firms
function draw_sample(data::DataFrame, id::Symbol)

    id_list = unique(data[!, id])
    
    ## Randomly chose observation
    # Generate vector with sample of IDs
    boot_choose = DataFrame(id => sample(id_list, length(id_list); replace=true))

    # Add unique identifier
    boot_choose.unique_id = range(1,length = length(boot_choose[!, id]))

    # Select obs in dataset that match the drawn numbers
    sample_data = innerjoin(data, boot_choose, on = id)

    return sample_data
end

function tj_print_res_bigestimator(data::DataFrame, Results::Results, Setup::Setup)
    
    # Production function parameters
    prd_fnc_array = hcat(
        string.(collect(keys(Results.point_estimates.prd_fnc))),
        collect(values(Results.point_estimates.prd_fnc)),
        collect(values(Results.std_errors.prd_fnc)),
        collect(values(Results.t_statistics.prd_fnc)),
        collect(values(Results.p_values.prd_fnc)),
        [ci[1] for ci in collect(values(Results.conf_intervals.prd_fnc))],  # CI lower
        [ci[2] for ci in collect(values(Results.conf_intervals.prd_fnc))]   # CI upper
    )

    # ω law-of-motion parameters
    lom_array = hcat(
        string.(collect(keys(Results.point_estimates.ω_lom))),
        collect(values(Results.point_estimates.ω_lom)),
        collect(values(Results.std_errors.ω_lom)),
        collect(values(Results.t_statistics.ω_lom)),
        collect(values(Results.p_values.ω_lom)),
        [ci[1] for ci in collect(values(Results.conf_intervals.ω_lom))],    # CI lower
        [ci[2] for ci in collect(values(Results.conf_intervals.ω_lom))]     # CI upper
    )

    headers = ["Estimate", "Std. Error", "t-statistic", "p-value", "CI Lower", "CI Upper"]
    println("")
    println("Observations: "*string(nrow(data)))
    println("Firms: "*string(length(unique(data[!, Setup.id]))))
    println("Bootstrap iterations: "*string(Setup.boot_reps))
    println("Final GMM criterion value: "*string(round(Results.criterion_value, digits=6)))
    pretty_table(vcat(prd_fnc_array, lom_array)[:,begin+1:end], column_labels = headers, formatters =  [fmt__printf("%5.5f")], row_labels = vcat(prd_fnc_array, lom_array)[:,1],
                 row_group_labels = [1 => "Production function parameters", 5 => "Ω law of motion parameters"], stubhead_label = "Variable",
                 row_group_label_alignment = :c, limit_printing = false)
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