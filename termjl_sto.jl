
using Plots
using Parameters
using Random
using Distributed
using Statistics
using DifferentialEquations

function CoralStoc!(du,u,p,t)
    C,M,H = u
    S = 1 - C - M
    @unpack i_c, b_c, alpha, d_c, i_m, b_m, g, n, r, sigma, f, event_rate, event_impact = p
    if rand() < event_rate
        du[1] -= event_impact * C
    else
        du[1] = (i_c + b_c*C)*S - d_c*C
    end
    du[2] = (i_m + b_m*M)*S - g*H*M
    du[3] = r*H*(1 - H) - f*H
end

# Define the Coral system with stochastic events
function CoralStocFeedback!(du, u, p, t)
    C, M, H = u
    @unpack i_c, b_c, alpha, d_c, i_m, b_m, g, n, r, sigma, f, event_rate, event_impact = p

    S = 1 - C - M

    if rand() < event_rate
        du[1] = - event_impact * C
        du[1] = - event_impact * M
        du[1] = - event_impact * H
    else
        du[1] = (i_c + b_c * C) * S * (1 - alpha * M) - d_c * C
        du[2] = (i_m + b_m * M) * S - g * H * M / (1 + g * n * M)
        du[3] = r * H * (1 - H / (1 - sigma + sigma * C)) - f * H
    end
    
    
    du[2] = (i_m + b_m * M) * S - g * H * M / (1 + g * n * M)
    du[3] = r * H * (1 - H / (1 - sigma + sigma * C)) - f * H

end


# Initial parameters and conditions
p_default = (i_c = 0.05,b_c=0.3,alpha=0.5,d_c=0.1,
i_m=0.05,b_m=0.8,g=1,n=1,r=1,sigma=0.6,f=0.4, 
event_rate = 0.1, event_impact = .5)
tspan = (0.0, 1000.0)
u0 = [0.7, 0.3, 0.5]

function parameter_sweep(param_name::Symbol, param_range)
    results = zeros(length(param_range)*11,4)
    a = 1

    for param_value in param_range
        for initial_cond in 0:10
            c = rand()
            m = abs(c - rand())
            h = rand()
            u0 = [c, m, h]
            p = merge(p_default, (param_name => param_value,))
            prob = ODEProblem(CoralStocFeedback!, u0, tspan, p)
            sol = solve(prob, Tsit5())
            results[a,2:4] = sol[1:3,end]
            results[a,1] = param_value
            a = a + 1
        end
    end
    return results
end



p_default = (i_c = 0.05,b_c=0.3,alpha=0.5,d_c=0.1,
i_m=0.05,b_m=0.8,g=1,n=1,r=1,sigma=0.6,f=0.3, 
event_rate = 0.19, event_impact = 0.5)

f_range = 0.1:0.01:1
tspan = (0.0, 500.0)

results = parameter_sweep(:f, f_range)
p4 = plot(xlabel="f", ylabel="Coral Abundance",title="Bifurcation with extreme event rate: $(p_default[12])")
scatter!(results[:, 1], results[:, 4],
 markersize=2, legend=false,color=:red)
display(p4)








#Monte Carlo 
# Initial parameters and conditions
p_default = (i_c = 0.05,b_c=0.3,alpha=0.5,d_c=0.1,
i_m=0.05,b_m=0.8,g=1,n=1,r=1,sigma=0.6,f=0.1, 
event_rate = 0.2, event_impact = .5)
tspan = (0.0, 500.0)
u0 = [0.7, 0.3, 0.5]
function monte_carlo_simulations(num_simulations::Int, param_defaults)
    final_coral_cover = zeros(num_simulations,3)
    for i in 1:num_simulations
        # Define the problem
        c = rand()
        m = abs(c - rand())
        h = rand()
        u0= [c,m,h]
        prob = ODEProblem(CoralStocFeedback!, u0, tspan, param_defaults)
        # Solve the problem
        sol = solve(prob, Tsit5())
        # Record the final coral cover
        final_coral_cover[i,:] = sol[1:3,end]
    end

    return final_coral_cover
end

num_simulations = 500
final_coral_cover = monte_carlo_simulations(num_simulations, p_default)

# Plot the histogram of final coral cover
histogram(final_coral_cover[:,1],title="n_sim = $(num_simulations), f = $(p_default[11])",
          xlabel="Final Coral Cover", ylabel="Frequency",
          label="rate = $(p_default[12]), impact = $(p_default[13])",bins=50)




#Ensemble Simu
prob = ODEProblem(CoralStoc!, u0, tspan, p_default)
function prob_func(prob, i, repeat)
    c = rand()
    m = abs(c - rand())
    h = rand()
    remake(prob, u0 = [c,m,h])
end
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 100)
plot(sim);




#Sensitivity EnsembleAnalysis
tspan = (0.0, 350.0)
u0 = [0.7, 0.3, 0.5]

# Define the parameter ranges for sensitivity analysis
param_ranges = Dict(
    :i_c => 0.01:0.01:0.06,
    :b_c => 0.1:0.1:0.4,
    :alpha => 0.3:0.1:0.6,
    :d_c => 0.05:0.05:0.15,
    :i_m => 0.04:0.01:0.06,
    :b_m => 0.6:0.2:1.0,
    :g => 0.8:0.2:1.2,
    :n => 0.8:0.2:1.2,
    :r => 0.8:0.2:1.2,
    :sigma => 0.4:0.2:0.8,
    :f => 0.1:0.1:0.3,
    :event_rate => 0.005:0.05:0.15,
    :event_impact => 0.5:0.1:0.9
)

# Perform sensitivity analysis
function sensitivity_analysis(p_default, param_ranges, num_simulations)
    results = Dict{Symbol, Vector{Float64}}()

    for (param, range) in param_ranges
        results[param] = Float64[]
        for param_value in range
            final_coral_cover = zeros(num_simulations)
            for i in 1:num_simulations
                # Generate random initial conditions
                c = rand()
                m = abs(c - rand())
                h = rand()
                u0 = [.7, .2, .5]

                # Update the parameter value
                p = merge(p_default, (param => param_value,))
                prob = ODEProblem(CoralStocFeedback!, u0, tspan, p)
                sol = solve(prob, Tsit5(), saveat=1.0)

                # Record the final coral cover
                final_coral_cover[i] = sol[end][1]
            end
            push!(results[param], mean(final_coral_cover))
        end
    end

    return results
end

# Run the sensitivity analysis
num_simulations = 100
results = sensitivity_analysis(p_default, param_ranges, num_simulations)


xlabel!("Parameter Value")
ylabel!("Mean Final Coral Cover")
title!("Sensitivity Analysis of Parameters")
display(plot())

first_six_params = collect(keys(param_ranges))[1:6]
p = plot(layout = (2, 3)) # create a 2x3 grid of subplots
for (i, param) in enumerate(first_six_params)
    plot!(p[i], param_ranges[param], results[param], title = string(param), label = false)
end
p