using Plots, DataFrames, Distributions, LinearAlgebra

#Problem 1 

function solve_basil(X, C, f, q, pmin, pmax, num_vendors)

    prices = pmin:0.1:pmax  #Possible prices
    n_prices = length(prices)  #Number of discrete price points
    n_vendors = num_vendors + 1  #Include n=0 state
    v = zeros(n_vendors)  #Value function
    v_approach = zeros(n_vendors)  #Value function for approaching a vendor
    σ_approach = zeros(Int, n_vendors)  #Policy for approaching vendors
    σ_buy = zeros(Int, n_vendors, n_prices)  #Policy for buying

    #The Bellman equation
    for n in n_vendors:-1:2
        
        v_terminate = -C * (n - 1)

        #Expected value of buying orchid at vendor n
        v_b = [X - p - C * (n - 1) for p in prices]

        #Expected value of approaching vendor n
        v_expected = sum(v_b) / n_prices  
        v_approach[n - 1] = -C * (n - 1) + f + q * v_expected + (1 - q) * v[n]

        #Optimal value
        v[n - 1] = max(v_terminate, v_approach[n - 1])

        σ_approach[n - 1] = v_approach[n - 1] > v_terminate ? 1 : 0

        for (i, p) in enumerate(prices)
            v_b_current = X - p - C * (n - 1)
            σ_buy[n - 1, i] = v_b_current > v[n] ? 1 : 0
        end
    end

    prob_buy = sum(q * σ_buy) / n_vendors  
    expected_price = sum(p * q * σ_buy for p in prices) / n_vendors 
    expected_vendors = sum(σ_approach) 

    return v, σ_approach, σ_buy, prob_buy, expected_price, expected_vendors
end

3. 
X = 50
C = 0.5
f = 1.0
q = 0.15
pmin = 10
pmax = 100
num_vendors = 50

v, σ_approach, σ_buy, prob_buy, expected_price, expected_vendors = solve_basil(X, C, f, q, pmin, pmax, num_vendors)

println("Probability Basil will buy the orchid: $prob_buy")
println("Expected price Basil will pay: $expected_price")
println("Expected number of vendors Basil will approach: $expected_vendors")


#Problem 2

β = 0.95         
c = 1.0            
w_min = 10.0      
dw = 1.0        
w_max = 100.0     
wages = w_min:dw:w_max
π_w = ones(length(wages)) / length(wages) 

p_vals = 0.0:0.05:1.0

reservation_wages = Float64[]
acceptance_probs = Float64[]
unemployment_durations = Float64[]

function solve_bellman(p)
    V_U = zeros(length(wages))  #Unemployed
    V_E = zeros(length(wages))  #Employed

    tol = 1e-6
    max_iter = 1000
    for _ in 1:max_iter
        V_U_old = copy(V_U)
        
        for i in eachindex(wages)
            w = wages[i]
            V_E[i] = w + β * ((1 - p) * V_E[i] + p * sum(π_w .* V_U))
        end

        for i in eachindex(wages)
            w = wages[i]
            V_U[i] = max(V_E[i], c + β * sum(π_w .* V_U))
        end

        #Checking convergence
        if maximum(abs.(V_U .- V_U_old)) < tol
            break
        end
    end

    reservation_wage = minimum(wages[V_E .>= c + β * sum(π_w .* V_U)])

    q = sum(π_w[wages .>= reservation_wage])

    expected_duration = 1 / q

    return reservation_wage, q, expected_duration
end

for p in p_vals
    w_star, q, duration = solve_bellman(p)
    push!(reservation_wages, w_star)
    push!(acceptance_probs, q)
    push!(unemployment_durations, duration)
end

#Plots
plot(p_vals, reservation_wages, label="Reservation Wage (w*)", xlabel="p (Job Separation Probability)", ylabel="w*", title="Reservation Wage vs p")
plot(p_vals, acceptance_probs, label="Acceptance Probability (q)", xlabel="p (Job Separation Probability)", ylabel="q", title="Acceptance Probability vs p")
plot(p_vals, unemployment_durations, label="Expected Unemployment Duration", xlabel="p (Job Separation Probability)", ylabel="Expected Duration", title="Unemployment Duration vs p")


#Problem 3

β = 0.95         
α = 0.3         
δ = 0.05        
γ_values = [0.5, 1, 2]  
k_star = ((β * α) / (1 - β * (1 - δ)))^(1 / (1 - α))  
k0 = 0.5 * k_star  

#Utility & Production functions
utility(c, γ) = γ == 1 ? log(c) : c^(1 - γ) / (1 - γ)  
production(k) = k^α 

function simulate_economy(γ, T=100)
    k_t = zeros(T)
    y_t = zeros(T)
    c_t = zeros(T)
    i_t = zeros(T)
    
    k_t[1] = k0
    for t in 1:T-1
        y_t[t] = production(k_t[t])
        c_t[t] = max(0.01, y_t[t] - δ * k_t[t] - (β^(1 / γ)) * (1 - δ) * k_t[t])  
        i_t[t] = y_t[t] - c_t[t]
        k_t[t + 1] = i_t[t] + (1 - δ) * k_t[t]  
    end
    y_t[T] = production(k_t[T])
    c_t[T] = y_t[T] - δ * k_t[T]
    i_t[T] = y_t[T] - c_t[T]
    return k_t, y_t, c_t, i_t
end


function compute_halfway_table(γ_values, k_star, k0, T=1000)
    table = []
    for γ in γ_values
        k_t, _, _, _ = simulate_economy(γ, T)
        halfway_index = findfirst(k -> abs(k_star - k) < 0.5 * (k_star - k0), k_t)
        push!(table, (γ, halfway_index))
    end
    return table
end

function plot_economy(γ_values, T=100)
    plt_k = plot(layout=(2, 2), legend=:topleft)
    plt_y = plot()
    plt_i = plot()
    plt_c = plot()
    
    for γ in γ_values
        k_t, y_t, c_t, i_t = simulate_economy(γ, T)
        plot!(plt_k, 1:T, k_t, label="γ=$γ", title="Capital (k)")
        plot!(plt_y, 1:T, y_t, label="γ=$γ", title="Output (y)")
        plot!(plt_i, 1:T, i_t ./ y_t, label="γ=$γ", title="Investment/Output")
        plot!(plt_c, 1:T, c_t ./ y_t, label="γ=$γ", title="Consumption/Output")
    end
    
    plot(plt_k, plt_y, plt_i, plt_c, layout=(2, 2), size=(800, 600))
end


T = 100
halfway_table = compute_halfway_table(γ_values, k_star, k0, T)
println("Halfway Table: γ and periods to close half the gap")
println(halfway_table)

plot_economy(γ_values, T)


#Problem 4

X = 0:5  
Z = 1:3 
P = [0.5 0.3 0.2;
     0.2 0.7 0.1;
     0.3 0.3 0.4] 

#Define σ(X_t, Z_t)
function σ(Xt, Zt)
    if Zt == 1
        return 0
    elseif Zt == 2
        return Xt
    elseif Zt == 3 && Xt <= 4
        return Xt + 1
    elseif Zt == 3 && Xt == 5
        return 3
    end
end


function compute_joint_transition_matrix(X, Z, P)
    nX, nZ = length(X), length(Z)
    T = zeros(nX * nZ, nX * nZ) 

    for xt_idx in 1:nX
        for zt_idx in 1:nZ
            current_state = (xt_idx - 1) * nZ + zt_idx 
            for zt_next_idx in 1:nZ
                Xt_next = σ(X[xt_idx], Z[zt_next_idx])  
                if Xt_next in X
                    xt_next_idx = findall(x -> x == Xt_next, X)[1]  
                    next_state = (xt_next_idx - 1) * nZ + zt_next_idx  
                    T[current_state, next_state] += P[zt_idx, zt_next_idx]
                end
            end
        end
    end

    return T
end

T = compute_joint_transition_matrix(X, Z, P)

#Stationary distribution
function stationary_distribution(T)
    eig = eigen(T')
    stationary = eig.vectors[:, argmax(eig.values .≈ 1)] 
    stationary ./= sum(stationary)  
    return stationary
end

stationary_dist = stationary_distribution(T)

function marginal_distribution_X(stationary_dist, nX, nZ)
    stationary_matrix = reshape(stationary_dist, nZ, nX) 
    marginal = sum(stationary_matrix, dims=1)  
    return marginal[:]
end

marginal_Xt = marginal_distribution_X(stationary_dist, length(X), length(Z))

#Expecvted Value
function expected_value_X(marginal_Xt, X)
    return sum(marginal_Xt .* X)
end

expected_X = expected_value_X(marginal_Xt, X)


println("Joint Transition Matrix T:")
println(T)

println("\nStationary Distribution of (X_t, Z_t):")
println(stationary_dist)

println("\nMarginal Distribution of X_t:")
println(marginal_Xt)

println("\nExpected Value of X_t:")
println(expected_X)
