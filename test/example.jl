using Pkg; Pkg.activate(pwd())
using TJProdEst, DataFrames, CSV, Optim, JLD2

# Read in data
data = load("test/example_data.jld2", "data")

# Basic example
res = tj_prod_est(data = data,
                  output = :Y,
                  flexible_input = :M,
                  fixed_inputs = [:K, :L],
                  flexible_input_price = :Pᴹ,
                  output_price = :Pʸ,
                  time = :year,
                  boot_reps = 10,
                  id = :ID);

# Example with more options
res = tj_prod_est(data = data,
                  output = :Y,
                  flexible_input = :M,
                  fixed_inputs = [:K, :L],
                  flexible_input_price = :Pᴹ,
                  output_price = :Pʸ,
                  ω_lom_degree = 1,
                  time = :year,
                  id = :ID,
                  boot_reps = 100,
                  std_err_estimation = true,
                  optimizer_options = (optimizer = NelderMead(), 
                                       startvals = [0.5, 0.5, 0.5, 0.4],
                                       optim_options = Optim.Options(iterations = 1000,
                                                        g_tol = 1e-8,
                                                        f_abstol = 1e-8,
                                                        x_abstol = 1e-8,
                                                        x_reltol = 1e-8,
                                                        allow_f_increases = true,
                                                        show_trace = false,
                                                        extended_trace = false,
                                                        show_every = 1,
                                                        time_limit = NaN,
                                                        store_trace = false)));

# Example with box constraints
res = tj_prod_est(data = data,
                  output = :Y,
                  flexible_input = :M,
                  fixed_inputs = [:K, :L],
                  flexible_input_price = :Pᴹ,
                  output_price = :Pʸ,
                  ω_lom_degree = 1,
                  time = :year,
                  id = :ID,
                  boot_reps = 100,
                  std_err_estimation = true,
                  optimizer_options = (optimizer = Fminbox(NelderMead()), 
                                       startvals = [0.6, 0.3, 0.4, 0.4], 
                                       lower_bound = [0.0, 0.0, 0.0, 0.0], 
                                       upper_bound = [1.0, 1.0, 1.0, 1.0],
                                       optim_options = Optim.Options(iterations = 1000,
                                                        g_tol = 1e-8,
                                                        allow_f_increases = true,
                                                        show_trace = false,
                                                        extended_trace = false,
                                                        show_every = 1,
                                                        time_limit = NaN,
                                                        store_trace = false)));