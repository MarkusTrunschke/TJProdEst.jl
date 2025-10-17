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
    display(tj_prodest_criterion(data = data, Setup = Setup, β = Setup.optimizer_options.startvals, weight = weights, c = c))
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
    if Results.criterion_value == Inf
        display(opt)
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
- `β::Vector{<:Number}`: candidate parameter vector. The first element is the
  flexible input coefficient (βᴹ) and the remaining are fixed input
  coefficients.
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

In-place computation of productivity (ω) terms and the law-of-motion (LOM) 
regression for the production function estimation. This function implements 
the core calculations for the proxy variable approach, computing current and 
lagged productivity, estimating the productivity LOM via OLS, and calculating 
the structural error (ξ).

This is a mutating function (hence the `!`) that updates the preallocated 
arrays in the `c` cache NamedTuple in-place for efficiency.

# Arguments
- `data::DataFrame`: The estimation dataset containing output, inputs, prices, 
  and lagged variables. Must include columns for current and lagged values of 
  all production function variables.
- `Setup::Setup`: The setup struct containing configuration parameters including 
  variable names, polynomial degree, and production function form.
- `β::Union{Vector{<:Number}, SubArray{...}}`: Production function parameters 
  vector. The ordering is: [constant, fixed_input_coeffs..., flexible_input_coeff].
  For Cobb-Douglas: β = [α₀, αₖ₁, αₖ₂, ..., αₘ] where K's are fixed inputs and 
  M is the flexible input.
- `c::NamedTuple`: Cache NamedTuple containing preallocated arrays for intermediate 
  calculations. Must include fields: `ω_array`, `ω_lom_array`, `ρ_hat`, `ξ_hat`. 
  These arrays are mutated in-place.

# Returns
- Nothing (the function mutates `c` in-place)

# Side Effects (Mutations)
The function updates the following fields in `c`:
- `c.ω_array`: Filled with current productivity (ω) computed from the production 
  function residual after accounting for inputs and the proxy variable.
- `c.ω_lom_array`: First column filled with lagged productivity (lag_ω), then 
  polynomial terms computed via `polynomial_fnc_fast!`, and ω-shifter columns 
  (if present) that were preallocated during setup.
- `c.ρ_hat`: Filled with OLS coefficients from regressing ω on its lagged 
  polynomial terms and ω-shifters (the LOM parameters).
- `c.ξ_hat`: Filled with structural error (innovation to productivity), computed 
  as ξ = ω - LOM(lag_ω, ω_shifters).

# Implementation Details
The productivity term ω is recovered from the production function by solving:
```
ω = ln(Y) + ln(Pᴹ·M / Pʸ·Y) - ln(M) - β₀ - βₘ - ∑βₖ·ln(K) - (βₘ - 1)·ln(M)
```
where the proxy variable relationship is used to invert for productivity.

The law-of-motion is estimated via OLS:
```
ω = ρ₀ + ρ₁·lag_ω + ρ₂·lag_ω² + ... + ρₚ·lag_ωᵖ + ρ_shifters·shifters + ξ
```

# Example
```julia
# Called internally during GMM criterion evaluation
tj_prod_reg!(est_data, Setup, β_current, cache)
# cache.ξ_hat now contains the structural errors
# cache.ρ_hat contains the LOM parameters
```

# See Also
- `tj_prodest_criterion`: Uses this function to compute moment conditions
- `polynomial_fnc_fast!`: Computes polynomial terms for the LOM
- `fastOLS`: Estimates the LOM coefficients
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

## Function calculating some SE related statistics
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

## Bootstrap function
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
            boot_data = draw_sample(data, Setup.id) # Draw bootstrap sample
            
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