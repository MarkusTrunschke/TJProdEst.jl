# TJProdEst

[![Build Status](https://github.com/MarkusTrunschke/TJProdEst.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MarkusTrunschke/TJProdEst.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package implements the production function estimation approach from [Trunschke and Judd (2024)](https://www.nber.org/papers/w33205), with a GMM estimator and bootstrap standard errors. Currently only one flexible input is supported. However, the package supports multiple fixed inputs and shifters in the law of motion of productivity.

Note that this is work in progress. Use at your own risk. Please report any issues to the author.

## Installation

Use Julia's integrated package manager [Pkg.jl](https://github.com/JuliaLang/Pkg.jl):

```julia
using Pkg
Pkg.add("TJProdEst")
```

## Usage

The main estimation function is `tj_prod_est`, which estimates production function parameters using a one-step GMM approach with optional bootstrap standard errors.

### Arguments

All arguments are keyword arguments. The required arguments are:

- `data` is a DataFrame containing all variables referenced by other arguments
- `output` is a Symbol representing the measure for firm's output
- `flexible_input` is a Symbol representing the measure for one flexible input (e.g., materials)
- `fixed_inputs` is a Symbol or Vector of Symbols representing of fixed production inputs (e.g., capital, labor)
- `flexible_input_price` is a Symbol for the flexible input price variable
- `output_price` is a Symbol for the output price variable
- `id` is a Symbol identifying each firm's unique identifier
- `time` is a Symbol identifying the observation's time period

### Optional Arguments

- `ω_lom_degree` is the polynomial degree for the productivity (ω) law of motion. Default is `1`.
- `ω_shifter` is a Symbol or Vector of Symbols for additional variables that shift productivity over time. Default is an empty vector.
- `std_err_estimation` is a Boolean indicating whether to compute standard errors. Default is `true`. It is intended to allow skipping the lengthly bootstrap estimation part if one wants to test specifications.
- `boot_reps` is an integer indicating the number of bootstrap replications for standard error estimation. Default is `200`
- `maximum_boot_tries` is the maximum number of retry attempts for failed bootstrap iterations. Default is `10`
- `optimizer_options` is a NamedTuple containing optimization settings:
  - `lower_bound` is a vector of lower bounds for parameters. Default is `-Inf` for all parameters. They do not have any effect if the optimizer is not of a `FminBox()` type.
  - `upper_bound` is a vector of upper bounds for parameters. Default is `Inf` for all parameters. They do not have any effect if the optimizer is not of a `FminBox()` type.
  - `startvals` is a vector of starting values for the optimization. Default is `0.5` for all parameters
  - `optimizer` specifies the optimization algorithm from [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/). Default is `Optim.NelderMead()`
  - `optim_options` is an `Optim.Options()` object for configuring the optimizer. See [Optim.Options](https://julianlsolvers.github.io/Optim.jl/stable/user/config/) for available options

### Returns

The function returns a NamedTuple `(Results, Setup)` where:
- `Results` is a mutable struct containing:
  - `point_estimates::NamedTuple`: Nested NamedTuple with `(prd_fnc, ω_lom)` containing parameter estimates for the production function and the law of motion.
  - `std_errors::NamedTuple`: Standard errors for all parameters
  - `variance::NamedTuple`: Variance estimates
  - `p_values::NamedTuple`: p-values for the t-test of the parameter being different from zero. 
  - `t_statistics::NamedTuple`: t-statistics for the t-test of the parameter being different from zero. 
  - `conf_intervals::NamedTuple`: Lower and upper bound of 95% confidence intervals
  - `criterion_value::Float64`: Final GMM criterion value at the point estimates.

- `Setup` is a struct containing the configuration used for estimation.

## Example

This example estimates a Cobb-Douglas production function with one flexible input (materials) and two fixed inputs (capital and labor).

```julia
# Load required dependencies
using TJProdEst, DataFrames, CSV

# Read in your panel data
# Assumes data has columns: ID, year, Y, M, K, L, Pᴹ, Pʸ
data = CSV.read("path/to/your/data.csv", DataFrame)

# Run estimation with default options
results, setup = tj_prod_est(
    data = data,
    output = :Y,
    flexible_input = :M,
    fixed_inputs = [:K, :L],
    flexible_input_price = :Pᴹ,
    output_price = :Pʸ,
    time = :year,
    id = :ID
)
```
Output looks like this (with your own results)
```julia
Progress: 100%[==================================================] Time: 0:00:03

Observations: 1900
Firms: 100
Bootstrap iterations: 200
Final GMM criterion value: 0.0
┌──────────┬──────────┬────────────┬─────────────┬─────────┬──────────┬──────────┐
│ Variable │ Estimate │ Std. Error │ t-statistic │ p-value │ CI Lower │ CI Upper │
├──────────┴──────────┴────────────┴─────────────┴─────────┴──────────┴──────────┤
│                         Production function parameters                         │
├──────────┬──────────┬────────────┬─────────────┬─────────┬──────────┬──────────┤
│ constant │  0.80490 │    0.06153 │    13.08067 │ 0.00000 │  0.68430 │  0.92551 │
│        K │  0.20201 │    0.02329 │     8.67471 │ 0.00000 │  0.15637 │  0.24765 │
│        L │  0.30615 │    0.01373 │    22.28972 │ 0.00000 │  0.27923 │  0.33307 │
│        M │  0.49980 │    0.00114 │   437.75606 │ 0.00000 │  0.49757 │  0.50204 │
├──────────┴──────────┴────────────┴─────────────┴─────────┴──────────┴──────────┤
│                           Ω law of motion parameters                           │
├──────────┬──────────┬────────────┬─────────────┬─────────┬──────────┬──────────┤
│        ω │  0.70824 │    0.03347 │    21.15993 │ 0.00000 │  0.64264 │  0.77384 │
└──────────┴──────────┴────────────┴─────────────┴─────────┴──────────┴──────────┘
```
### Accessing results
```
# Access production function parameters
results.point_estimates.prd_fnc
# Output: (constant = 0.123, K = 0.456, L = 0.321, M = 0.234)

# Access productivity law-of-motion parameters
results.point_estimates.ω_lom
# Output: (ω = 0.789)

# View standard errors
results.std_errors.prd_fnc
# Output: (constant = 0.045, K = 0.067, L = 0.054, M = 0.032)

# View confidence intervals
results.conf_intervals.prd_fnc
# Output: (constant = [0.035, 0.211], K = [0.325, 0.587], ...)
```

### Example with Custom Options

```julia
using Optim

# Estimate with box constraints and custom optimizer settings
results, setup = tj_prod_est(
    data = data,
    output = :Y,
    flexible_input = :M,
    fixed_inputs = [:K, :L],
    flexible_input_price = :Pᴹ,
    output_price = :Pʸ,
    ω_lom_degree = 2,  # Quadratic productivity law of motion
    time = :year,
    id = :ID,
    boot_reps = 500,  # More bootstrap replications
    optimizer_options = (
        lower_bound = [0.0, 0.0, 0.0, 0.0],  # Non-negative coefficients
        upper_bound = [1.0, 1.0, 1.0, 1.0],  # Upper bounds
        startvals = [0.1, 0.3, 0.3, 0.3],
        optimizer = Fminbox(NelderMead()),  # Box-constrained optimizer
        optim_options = Optim.Options(
            iterations = 10000,
            show_trace = true
        )
    )
)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Author

* **Markus Trunschke:** mtrunsch@stanford.edu

## License

Please cite this package and the [paper the estimator is based on](https://www.nber.org/papers/w33205) if you use it for published research.

[MIT](https://choosealicense.com/licenses/mit/)
