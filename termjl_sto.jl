
using Plots
using Parameters
using Random
using Distributed
using Statistics
using DifferentialEquations


# Define the coral system without multiple feedback systems
function CoralStoc!(du,u,p,t)
    C,M,H = u
    S = 1 - C - M
    @unpack i_c, b_c, alpha, d_c, i_m, b_m, g, n, r, sigma, f, event_rate, event_impact = p
    if rand() < event_rate
        du[1] = - event_impact * C
        du[2] = - event_impact * M
        du[3] = - event_impact * H
    else
        du[1] = (i_c + b_c*C)*S - d_c*C
        du[2] = (i_m + b_m*M)*S - g*H*M
        du[3] = r*H*(1 - H) - f*H
    end
    
   
end

# Define the Coral system with multiple feedbacks and with stochastic events
function CoralStocFeedback!(du, u, p, t)
    C, M, H = u
    @unpack i_c, b_c, alpha, d_c, i_m, b_m, g, n, r, sigma, f, event_rate, event_impact = p

    S = 1 - C - M

    if rand() < event_rate
        du[1] = - event_impact * C
        du[2] = - event_impact * M
        du[3] = - event_impact * H
    else
        du[1] = (i_c + b_c * C) * S * (1 - alpha * M) - d_c * C
        du[2] = (i_m + b_m * M) * S - g * H * M / (1 + g * n * M)
        du[3] = r * H * (1 - H / (1 - sigma + sigma * C)) - f * H
    end
end


# Initial parameters and conditions
p_default = (i_c = 0.05,b_c=0.3,alpha=0.5,d_c=0.1,
i_m=0.05,b_m=0.8,g=1,n=1,r=1,sigma=0.6,f=0.4, 
event_rate = 0.1, event_impact = .5)
tspan = (0.0, 1000.0)
u0 = [0.7, 0.3, 0.5]

#Functions to sweep the parameters and create a bifurcation plot
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


#Monte Carlo Simulations of the stohcastic model
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
