"""
Auxiliary functions and result container for TJProdEst.

Define the result struct at top-level (required by Julia) and provide a
constructor function `res_struct_init()` that returns a default instance.
"""

"""
    Results

Mutable struct storing estimation results with nested NamedTuples for production function and ω law-of-motion parameters.

# Fields
- `point_estimates::NamedTuple`: Point estimates with `(prd_fnc, ω_lom)` structure
- `std_errors::NamedTuple`: Standard errors 
- `variance::NamedTuple`: Variance estimates
- `p_values::NamedTuple`: p-values for hypothesis tests
- `t_statistics::NamedTuple`: t-statistics
- `conf_intervals::NamedTuple`: Confidence intervals (2-element vectors)
- `criterion_value::Float64`: Final GMM criterion value
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

Immutable struct storing estimation configuration and data settings.

# Fields
- `output::Symbol`: Output variable column name
- `flexible_input::Symbol`: Flexible input variable (e.g., materials)
- `fixed_inputs::Union{Symbol,Vector{Symbol}}`: Fixed input(s) (e.g., capital, labor)
- `flexible_input_price::Symbol`: Price for flexible input
- `all_inputs::Vector{Symbol}`: Concatenation of `fixed_inputs` and `flexible_input`
- `output_price::Symbol`: Output price variable
- `ω_lom_degree::Int`: Polynomial degree for ω law-of-motion
- `ω_shifter::Union{Symbol, Vector{Symbol}}`: Additional shifters in ω LOM
- `time::Union{Symbol, Missing}`: Time variable for panel data
- `id::Union{Symbol, Missing}`: Firm/panel identifier
- `prd_fnc_form::String`: Production function form (e.g., "CobbDouglas")
- `std_err_estimation::Bool`: Whether to compute standard errors
- `std_err_type::String`: Standard error method (e.g., "Bootstrap")
- `boot_reps::Int`: Number of bootstrap replications
- `maximum_boot_tries::Int`: Maximum retry attempts for bootstrap
- `optimizer_options::NamedTuple`: Optimization settings (bounds, startvals, optimizer, optim_options)
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

Initialize a `Results` struct with `missing` values based on `Setup` configuration. Creates nested NamedTuples for production function (constant + all_inputs) and ω law-of-motion (ω terms + shifters) parameters.
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

Construct a `Setup` struct with all estimation configuration. Merges user-provided `optimizer_options` 
    with defaults and builds `all_inputs` by concatenating `fixed_inputs` and `flexible_input`.

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

"""
    jt_data_prep(data::DataFrame, Setup::Setup) -> DataFrame

Prepare panel data for estimation by computing proxy variable (log flexible input share) and creating lagged variables. Returns filtered dataset with complete observations.
"""
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
    panel_lag!(; data, id, time, variable, lag_prefix="lag_", lags=1, drop_missings=false, force=false) -> DataFrame

Compute panel lags in-place using `ShiftedArrays.lag` within groups. Sorts by `id` and `time`, creates lag columns with specified prefix, and validates time gaps.

# Keyword Arguments
- `data::DataFrame`: Data frame to mutate
- `id::Symbol`: Panel identifier column
- `time::Symbol`: Time variable for ordering
- `variable::Union{Vector{Symbol},Symbol}`: Column(s) to lag
- `lag_prefix::String="lag_"`: Prefix for lag column names
- `lags::Int=1`: Lag distance
- `drop_missings::Bool=false`: Drop rows with missing lags
- `force::Bool=false`: Remove existing lag columns if present
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
    lagging_that_panel!(; data, id, time, variable, lag_prefix="lag_", lags=1, drop_missings=false) -> DataFrame

Internal helper for `panel_lag!`. Computes lags using `ShiftedArrays.lag` within groups, validates time gaps (sets to `missing` if gap ≠ `lags`), joins back to data, and renames columns.
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

In-place computation of polynomial terms. Fills columns 2 through `degree` of `poly_mat` with powers 2 through `degree` of column 1. Preallocated matrix avoids allocations during repeated calls.

# Arguments
- `poly_mat::Union{Array{<:Number}, SubArray{<:Number}}`: Matrix with base values in column 1
- `degree::Int`: Highest polynomial degree to compute

# Keyword Arguments
- `par_cal::Bool=false`: Use `Threads.@threads` for parallel computation
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

Compute OLS coefficients β̂ = (X'X)⁻¹X'Y using in-place Cholesky decomposition for minimal allocations. Optionally checks and handles multicollinearity.

# Keyword Arguments
- `Y::Union{Matrix, Vector}`: Dependent variable(s)
- `X::Union{Matrix{<:Number}, Vector{<:Number}}`: Design matrix of regressors
- `multicolcheck::Bool=true`: Check for perfect multicollinearity
- `force::Bool=false`: Auto-drop multicollinear columns with warning instead of error
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

"""
    draw_sample(; data, id, sample_size=length(unique(data[!, id])), with_replacement=true) -> DataFrame

Draw bootstrap sample of firms from panel data. Assigns unique IDs to resampled firms to avoid duplicate ID issues in subsequent panel operations.

# Keyword Arguments
- `data::DataFrame`: Panel dataset to sample from
- `id::Symbol`: Firm identifier column
- `sample_size::Int`: Number of firms to sample (default: all unique firms)
- `with_replacement::Bool=true`: Sample with or without replacement
"""
function draw_sample(;data::DataFrame, id::Symbol, sample_size::Int=length(unique(data[!, id])), with_replacement::Bool=true)
    
    ## Randomly chose observation
    # Generate vector with sample of IDs
    boot_choose = DataFrame(id => sample(unique(data[!, id]), sample_size, replace=with_replacement))

    # Add unique identifier
    boot_choose.unique_id = range(1,length = length(boot_choose[!, id]))

    # Select obs in dataset that match the drawn numbers
    sample_data = innerjoin(data, boot_choose, on = id)

    return sample_data
end

"""
    tj_print_res_bigestimator(data::DataFrame, Results::Results, Setup::Setup)

Print formatted estimation results table using `PrettyTables`. Displays production function and ω law-of-motion parameters with standard errors, t-statistics, p-values, and confidence intervals.
"""
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
    superscript_this!(c::String) -> Char

Convert first character of string to its Unicode superscript equivalent using `superscript_map`. Returns original character if no superscript exists.
"""
function superscript_this!(c::String) # Need to use a string as input because I don't understand Chars in Julia. Char(5) returns a different unicode than string(5). And the superscript of Char(5) does not  work
    # Return the superscript character if it exists in the map, else return the original character
    return get(superscript_map, c[1], c[1])
end