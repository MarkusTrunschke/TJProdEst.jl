"""
    tj_prod_est(; data, output, flexible_input, fixed_inputs, flexible_input_price, output_price, ω_lom_degree=1, ω_shifter=[], time, id, std_err_estimation=true, std_err_type="Bootstrap", boot_reps=200, maximum_boot_tries=10, optimizer_options=NamedTuple())

Top-level estimation entry point for production function estimation using the
approach described in Trunschke and Judd (2024). Returns a tuple `(Results, Setup)` with parameter
estimates and configuration.

# Keyword Arguments
- `data::DataFrame`: Input dataset with (firm-time) panel structure
- `output::Symbol`: Output variable column name
- `flexible_input::Symbol`: Flexible input variable (e.g., materials)
- `fixed_inputs::Union{Symbol,Vector{Symbol}}`: Fixed input variable(s) (e.g., capital, labor)
- `flexible_input_price::Symbol`: Price of flexible input
- `output_price::Symbol`: Output price
- `ω_lom_degree::Int=1`: Polynomial degree for productivity law-of-motion
- `ω_shifter::Union{Symbol,Vector{Symbol}}=[]`: Optional productivity shifter variables
- `time::Symbol`: Time period identifier
- `id::Symbol`: Firm/unit identifier
- `std_err_estimation::Bool=true`: Whether to compute standard errors
- `std_err_type::String="Bootstrap"`: Type of standard errors ("Bootstrap" only currently)
- `boot_reps::Int=200`: Number of bootstrap replications
- `maximum_boot_tries::Int=10`: Max retry attempts per failed bootstrap iteration
- `optimizer_options::NamedTuple=NamedTuple()`: Optimization settings (see Optim.jl)

# Returns
- `NamedTuple`: `(Results, Setup)` containing estimates and configuration

# Example
```julia
results = tj_prod_est(
    data = df,
    output = :Y,
    flexible_input = :M,
    fixed_inputs = [:K, :L],
    flexible_input_price = :Pᴹ,
    output_price = :Pʸ,
    time = :year,
    id = :firm_id,
)
```
"""
function tj_prod_est(;data::DataFrame, 
                      output::Symbol, 
                      flexible_input::Symbol, 
                      fixed_inputs::Union{Symbol,Vector{Symbol}}, 
                      flexible_input_price::Symbol, 
                      output_price::Symbol, 
                      ω_lom_degree::Int=1, 
                      ω_shifter::Union{Symbol, Vector{Symbol}} = Vector{Symbol}(undef,0), 
                      time::Symbol, 
                      id::Symbol, 
                      std_err_estimation::Bool = true, 
                      std_err_type::String = "Bootstrap",
                      boot_reps::Int = 200,
                      maximum_boot_tries::Int = 10,
                      optimizer_options::NamedTuple=NamedTuple())

    prd_fnc_form = "CobbDouglas"

    # Unify fixed_inputs to always be a Vector
    if isa(fixed_inputs, Symbol)
        fixed_inputs = [fixed_inputs]
    end
    if isa(ω_shifter, Symbol)
        ω_shifter = [ω_shifter]
    end

    # Initialize struct to hold setup parameters
    Setup = setup_struct_init(output, flexible_input, fixed_inputs, flexible_input_price, output_price, 
                              ω_lom_degree, ω_shifter, time, id, prd_fnc_form, std_err_estimation, std_err_type, 
                              boot_reps, maximum_boot_tries, optimizer_options)

    # Initialize struct to hold results and meta data
    Results = res_struct_init(Setup)

    # Set up data
    est_data = jt_data_prep(data, Setup)

    # Estimate parameters
    tj_prodest_estimation!(est_data, Setup, Results)

    # Print results
    tj_print_res_bigestimator(est_data, Results, Setup)

    return (Results = Results, Setup = Setup)
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
function tj_prodest_estimation!(data::DataFrame, Setup::Setup, Results::Results)

    # Update Results with new point estimates
    Results.point_estimates = tj_onestep_estimator(data, Setup, Results)
    if Setup.std_err_estimation
        tj_std_error_stats(data, Setup, Results)
    end
end

"""
    tj_onestep_estimator(data, Setup, Results) -> NamedTuple

Perform one-step GMM estimation of production function parameters.

# Arguments
- `data::DataFrame`: Prepared estimation dataset containing output, inputs,
  prices, and lagged variables. Should be the output of `jt_data_prep`.
- `Setup::Setup`: Configuration struct containing model specification (variable
  names, polynomial degree, optimizer settings, etc.).
- `Results::Results`: Results struct that will store the criterion value and
  is used to determine the structure of output estimates.

# Returns
- `NamedTuple`: A nested NamedTuple with two fields:
  - `prd_fnc`: Production function parameters as a NamedTuple with keys like
    `:constant`, `:K`, `:L`, `:M` (depends on `Setup.all_inputs`)
  - `ω_lom`: Productivity law-of-motion parameters as a NamedTuple with keys
    like `:ω`, `:ω²`, etc. (depends on `Setup.ω_lom_degree` and `Setup.ω_shifter`)

# Optimization Details
The optimization uses `Optim.jl`'s numerical optimization routines and allows to set
all optimizer options with `Optim.Options(...)`. One can set Box constraints
providing the `lower_bound` and `upper_bound` arguments in `TJProdEst.tj_prod_est` combined
with a `Optim.Fminbox` optimizer.

# Throws
- Throws an error with message "Estimation did not converge..." if the
  optimizer fails to converge.
- Throws an error for unsupported production function forms (currently only
  "CobbDouglas" is supported).
"""
function tj_onestep_estimator(data::DataFrame, Setup::Setup, Results::Results)

    # Define weights as an identity matrix. Weighting does not matter (exactly identified)
    weights = I

    # Preallocate cache
    c = (
        ω_array = Array{Float64}(undef, size(data,1),1),
        lag_ω_poly_array = Array{Float64}(undef, size(data,1), Setup.ω_lom_degree),
        ϵ = Array{Float64}(undef, size(data,1),1),
        ξ_hat = Array{Float64}(undef, size(data,1),1),
        ρ_hat =  Vector{Float64}(undef, Setup.ω_lom_degree + length(Setup.ω_shifter)),
        m_mat = Array{Float64}(undef, length(Results.point_estimates.prd_fnc), 1),
        ω_lom_array = Array{Float64}(undef, size(data,1), Setup.ω_lom_degree + length(Setup.ω_shifter))
    )

    # Can already allocate ω_lom_array colums for ω shifter here (they will not change during optimization)
    if length(Setup.ω_shifter) != 0
        c.ω_lom_array[:,end-length(Setup.ω_shifter)+1:end] .= data[!, Setup.ω_shifter]
    end

    # Optimize
    opt = if typeof(Setup.optimizer_options.optimizer) <: Fminbox
        optimize(par -> tj_prodest_criterion(data = data, Setup = Setup, β = par, weight = weights, c = c),
                                             Setup.optimizer_options.lower_bound, Setup.optimizer_options.upper_bound,
                                             Setup.optimizer_options.startvals, Setup.optimizer_options.optimizer,
                                             Setup.optimizer_options.optim_options)
    else
        optimize(par -> tj_prodest_criterion(data = data, Setup = Setup, β = par, weight = weights, c = c),
                                             Setup.optimizer_options.startvals, Setup.optimizer_options.optimizer,
                                             Setup.optimizer_options.optim_options)
    end
    
    # Throw an error message if GMM did not converge
    if !Optim.converged(opt)
        throw("Estimation did not converge. Check data and numerical optimizer options.")
    end

    prd_fnc_estimates = if Setup.prd_fnc_form == "CobbDouglas"
        (; Pair.(vcat(:constant, Setup.all_inputs), Optim.minimizer(opt))...)
    else
        throw("Unsupported production function form: "*Setup.prd_fnc_form)
    end
     
    # Calculate ω lom parameters
    tj_prod_reg!(data, Setup, Optim.minimizer(opt), c)

    if Results.criterion_value == Inf
        Results.criterion_value = Optim.minimum(opt)
    end

    return (prd_fnc = prd_fnc_estimates, ω_lom = (; Pair.(Symbol.(keys(Results.point_estimates.ω_lom)), c.ρ_hat)...))
end

"""
    tj_prodest_criterion(; data, Setup, β, weight, c)

Compute the GMM (Generalized Method of Moments) criterion function value for
the production function estimation. This function evaluates the weighted
squared sum of moment conditions given a candidate parameter vector `β`.

The criterion is minimized during optimization to find the parameter estimates
that best satisfy the moment conditions (orthogonality between instruments
and residuals).

# Keyword arguments
- `data::DataFrame`: prepared estimation dataset with lagged variables and
  transformed columns.
- `Setup::Setup`: configuration struct containing model specification (inputs,
  variables, degree of ω polynomial, etc.).
- `β::Vector{<:Number}`: candidate parameter vector ordered as 
  [constant, fixed_input_coeffs..., flexible_input_coeff].
- `weight::Union{Array,UniformScaling}`: weighting matrix for the moment
  conditions. Often set to identity matrix `I` for exactly identified models.
- `c::NamedTuple`: preallocated cache containing arrays for intermediate
  calculations (ϵ, ξ_hat, ρ_hat, m_mat, ω_array, etc.) to avoid repeated
  allocations.

# Returns
- `Float64`: the GMM criterion value (weighted sum of squared moments),
  scaled by sample size.

# Notes
- The function computes moment conditions based on orthogonality between
  productivity shocks (ϵ, ξ) and observables (proxy variable, fixed inputs).
- Calls `tj_prod_reg!` internally to compute residuals and ω law-of-motion
  parameters.
- The criterion is minimized by the optimizer in `tj_onestep_estimator`.
"""
function tj_prodest_criterion(;data::DataFrame, Setup::Setup, β::Vector{<:Number}, weight::Union{Array{<:Number},LinearAlgebra.UniformScaling{Bool}}, c::NamedTuple)
    # Get ϵ
    c.ϵ .= log(β[end]) .- data[!, :ln_proxy_output_frac]
    
    # Get ξ and ρ (outsourced in another program b/c it is used in multiple locations)
    tj_prod_reg!(data, Setup, β, c)
    
    # Moments for scaling parameter and fixed inputs
    c.m_mat[begin:end-1] .= vec(c.ξ_hat' * Array(hcat(data[:, :constant], log.(data[!, Setup.fixed_inputs]))))

    # Moment for flexible input
    c.m_mat[end] = (c.ϵ' * log.(data[!, Setup.flexible_input]))[1]

    # Return criterion
    return (c.m_mat' * weight * c.m_mat)[1]
end

"""
    tj_prod_reg!(data, Setup, β, c) -> Nothing

Compute productivity (ω) and its law-of-motion via in-place OLS regression.
Mutates the cache `c` with productivity terms, LOM parameters, and structural
errors.

# Arguments
- `data::DataFrame`: Estimation dataset with output, inputs, and lagged variables
- `Setup::Setup`: Configuration (variable names, polynomial degree)
- `β::Vector{<:Number}`: Production function parameters [constant, fixed_inputs..., flexible_input]
- `c::NamedTuple`: Preallocated cache to mutate

# Returns
- `Nothing`: The function mutates the cache `c` in-place.

# Side Effects
Updates `c` fields:
- `ω_array`: Current productivity
- `ω_lom_array`: Lagged productivity and polynomial terms
- `ρ_hat`: LOM coefficients (from OLS of ω on lag_ω polynomials + shifters)
- `ξ_hat`: Productivity innovations (ω - predicted LOM)
"""
function tj_prod_reg!(data::DataFrame, Setup::Setup,
                       β::Union{Vector{<:Number},SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}},
                       c::NamedTuple = ())

    # Get ω
    c.ω_array .= log.(data[:, Setup.output]) .+ data[!, :ln_proxy_output_frac] .- log.(data[:, Setup.flexible_input]) .- 
                 log.(@view(β[begin])) .- log.(@view(β[end])) .- log.(Array(data[:, Setup.fixed_inputs])) * @view(β[begin+1:end-1]) .- 
                 (@view(β[end]) .- 1) .* log.(data[:, Setup.flexible_input])

    c.ω_lom_array[:,1] .= log.(data[!, Symbol("lag_", Setup.output)]) .+ data[!, Symbol("lag_", :ln_proxy_output_frac)] .- 
                               log.(data[!, Symbol("lag_", Setup.flexible_input)]) .- 
                               log.(@view(β[begin])) .- log.(@view(β[end])) .- log.(Array(data[!, Symbol.("lag_", Setup.fixed_inputs)])) * 
                               β[begin+1:end-1] .- (@view(β[end]) .- 1) .* log.(data[!, Symbol("lag_", Setup.flexible_input)])

    # Copy lagged ω to ω_lom_array first column, then calculate polynomials from lag_ω_hat
    polynomial_fnc_fast!(@view(c.ω_lom_array[:,1:Setup.ω_lom_degree]), Setup.ω_lom_degree, par_cal = false)

    # Get ξ by regressing omega onto omega_lag and ω-shifters
    c.ρ_hat .= fastOLS(Y = c.ω_array, X = c.ω_lom_array)

    # Define residual being the difference b/w omega, lagged omega and the ω-shifters
    c.ξ_hat .= c.ω_array .- c.ω_lom_array * c.ρ_hat
end

"""
    tj_std_error_stats(data, Setup, Results) -> Nothing

Compute standard errors, t-statistics, p-values, and confidence intervals via
bootstrap resampling. Mutates the `Results` struct in-place with statistical
inference results.

# Arguments
- `data::DataFrame`: Estimation dataset
- `Setup::Setup`: Configuration including bootstrap settings (`boot_reps`, `std_err_type`)
- `Results::Results`: Results struct to update with inference statistics

# Side Effects
Populates the following fields in `Results`:
- `variance`: Bootstrap variance estimates
- `std_errors`: Standard errors (√variance)
- `t_statistics`: t-statistics for hypothesis testing
- `p_values`: Two-sided p-values (assuming normality)
- `conf_intervals`: 95% confidence intervals (±1.96 × SE)

# Notes
- Currently only supports bootstrap standard errors
- Uses `bootstrap_tj_prodest` to generate bootstrap samples
"""
function tj_std_error_stats(data::DataFrame, Setup::Setup, Results::Results)

    if Setup.std_err_type == "Bootstrap"
        var_vec = vec(var(bootstrap_tj_prodest(data, Setup, Results), dims = 1)) # Run Bootstrap repetitions
        
        # Store as NamedTuples matching the structure of point_estimates
        Results.variance = (
            prd_fnc = (; Pair.(keys(Results.point_estimates.prd_fnc), var_vec[begin:length(Results.point_estimates.prd_fnc)])...),
            ω_lom = (; Pair.(keys(Results.point_estimates.ω_lom), var_vec[length(Results.point_estimates.prd_fnc)+1:end])...),
        )
    else
        throw("Only Bootstrap standard errors are currently implemented.")
    end

    # Calculate some quantities
    std_errors = sqrt.(var_vec)
 
    ## Inference (see Cameron and Trivado (2005) ch.11)
    point_est = vcat(collect(values(Results.point_estimates.prd_fnc)), collect(values(Results.point_estimates.ω_lom)))

    # Calculate t-statistic
    tstat_vec = point_est ./ std_errors

    # P-values (using N(0,1))
    p_values = 2 .*(1 .- cdf.(Normal(),abs.(tstat_vec)))

    # Confidence intervals (using N(0,1))
    conf_int_vec = hcat(point_est - 1.96.*std_errors, point_est + 1.96.*std_errors)

    # Store all results in Reults struct
    Results.std_errors = (
            prd_fnc = (; Pair.(keys(Results.point_estimates.prd_fnc), std_errors[begin:length(Results.point_estimates.prd_fnc)])...),
            ω_lom = (; Pair.(keys(Results.point_estimates.ω_lom), std_errors[length(Results.point_estimates.prd_fnc)+1:end])...),
        )
    Results.t_statistics = (
            prd_fnc = (; Pair.(keys(Results.point_estimates.prd_fnc), tstat_vec[begin:length(Results.point_estimates.prd_fnc)])...),
            ω_lom = (; Pair.(keys(Results.point_estimates.ω_lom), tstat_vec[length(Results.point_estimates.prd_fnc)+1:end])...),
        )
    Results.p_values = (
            prd_fnc = (; Pair.(keys(Results.point_estimates.prd_fnc), p_values[begin:length(Results.point_estimates.prd_fnc)])...),
            ω_lom = (; Pair.(keys(Results.point_estimates.ω_lom), p_values[length(Results.point_estimates.prd_fnc)+1:end])...),
        )
    Results.conf_intervals = (
            prd_fnc = (; Pair.(keys(Results.point_estimates.prd_fnc), eachrow(conf_int_vec[begin:length(Results.point_estimates.prd_fnc),:]))...),
            ω_lom = (; Pair.(keys(Results.point_estimates.ω_lom), eachrow(conf_int_vec[length(Results.point_estimates.prd_fnc)+1:end,:]))...),
        )
    
    
end

"""
    bootstrap_tj_prodest(data, Setup, Results) -> Matrix{Float64}

Generate bootstrap estimates by resampling firms and re-estimating the model.
Returns a matrix where each row contains parameter estimates from one bootstrap
replication.

# Arguments
- `data::DataFrame`: Estimation dataset
- `Setup::Setup`: Configuration with `boot_reps` and `maximum_boot_tries`
- `Results::Results`: Results struct (used for parameter structure)

# Returns
- `Matrix{Float64}`: Bootstrap estimates matrix of size 
  `(boot_reps × n_params)` where n_params = production function params + 
  ω LOM params. Each row is one bootstrap replication.

# Details
- Uses panel bootstrap: samples firms with replacement, keeps entire time series
- Parallelized across bootstrap repetitions using `Threads.@threads`
- Retries failed estimations up to `maximum_boot_tries` times per replication
"""
function bootstrap_tj_prodest(data::DataFrame, Setup::Setup, Results::Results)
    
    # Set some counters
    rep = 1
    ntry = 1

    # Preallocate cache
    res_vec = Array{Float64}(undef, Setup.boot_reps, length(Results.point_estimates.prd_fnc) + length(Results.point_estimates.ω_lom))

    # Run bootstrap iterations
    p = Progress(Setup.boot_reps; dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:white)
    Threads.@threads for rep = 1:Setup.boot_reps

        successfull_rep = false # Define exit flag

        while successfull_rep == false
            boot_data = draw_sample(data=data, id=Setup.id, with_replacement=true) # Draw bootstrap sample
            
            # Estimate model parameters with bootstrap sample
            try
                boot_res = tj_onestep_estimator(boot_data, Setup, Results)
                res_vec[rep,:] .= vcat(collect(values(boot_res[1])), collect(values(boot_res[2])))
            catch
                if ntry < Setup.maximum_boot_tries
                    ntry += 1
                    continue # Repeat iteration with new sample
                else
                    throw("Error: Maximal number of bootstrap iteration tries reached at bootstrap repetition "*string(rep))
                end
            end
            successfull_rep = true # Set exit flag to true
        end
        next!(p)
    end
    finish!(p)
    return res_vec
end