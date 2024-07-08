#=
Callbacks.jl

File contents:

Subfunctions necessary to construct the callbacks used in the dynamic integration of a square waveform catalyst oscillation. 

Functions included: 
    build_callbacks: Creates the callbacks used to implement square-waveform dynamics and check for Steady State
        Once steady state is reached (marked by the alignment of the surface coverages for the previous oscillation with one 5 oscillations back) the integration is terminated 
    --> Returns: switch_times (array with the timepoints for square wave form callbacks), cbs_switches (Callback sets of each discrete callback)
    
    calc_rates: calculates instantaneous elementary rates & vector of catalyst state (Tracks the rates throughout the oscillation)
    --> Returns: elem_rates_inst (array contained the instantaneous rates throughout the oscillation), state_vector (an array of 1 and 2 indicating the catalytic state at each point in the solution)
    
    find_switches: Maps state switch times to the solution (sol.t) ODE solver output indices
    --> Returns: switch_indices (marks the indices of the solution vectors where the catalyst state switches)
        
    cutoff: cuts the the switch_times to match the integration length
    --> Returns: t_cutoff (the time the integration ends) and switches (the array of switch times spliced at t_cutoff)
    
    write_to_csv: takes in a filepath and a data frame with results and writes the data frame to the csv in the filepath
    --> Returns: nothing

    define_loop: checks if all elementary rates are equal and returns the appropriate loop TOF 
    --> Returns: loopTOF which equals zero in the absence of a complete loop and equals the average of the elementary rates in the case that they are all equivalent

Written in Julia 1.9.0

Author: Madeline Murphy
=#

## Build Callbacks
function build_callbacks(tau,Numosc,k)
## Creates the callbacks used to implement square-wave dynamic 
    
    # Timepoints for callbacks
    switch_S2 = [n for n in tau[1]:sum(tau):Numosc*sum(tau)];
    switch_S1 = [n for n in sum(tau):sum(tau):Numosc*sum(tau)];
    switch_times = sort!(vcat(switch_S1,switch_S2));
    SS_check_times = [n for n in 20*sum(tau):20*sum(tau):Numosc*sum(tau)];
    
    # Initiate the conditions to reach SS
    cond1 = 1;
    cond2 = 1;
    cond3 = 1;
    num_osc = 0;
    
    # Build callback functions
    # Conditions
    function condition_S2(u,t,integrator)
        t ∈ switch_S2
    end
    function condition_S1(u,t,integrator)
        t ∈ switch_S1
    end
    
    # Affects
    function affect_S2!(integrator)
        integrator.p = big.(k[:,2])
    end
    function affect_S1!(integrator)
        integrator.p = big.(k[:,1])
    end

    # Condtion to reach steady state --> checks for steady-state every 10 oscillation through comparison of the time-averaged rates
    function condition_steady_state(u,t,integrator)
        if t ∈ SS_check_times
            # get the current time
            t_end = integrator.t
            # determine the number of oscillations and cutoff the switch_times vector
            num_osc = floor(Int, (t_end/sum(tau)));
            switches = switch_times[1:2*num_osc-2]
            
            # find the indices of state switches
            switch_indices = find_switches(switches, integrator.sol.t)
            
            
            # isolate the time array and the solution array for the surface coverages
            t_array = integrator.sol.t;
            x_inst = transpose(hcat((integrator.sol.u)...));
            
            # compute the elementary rates vector 
            r_inst, state_vector, k_inst = calc_rates(t_array,x_inst,switch_indices,k)
            
            # compute time averaged rates over the past 2 oscillation
            idx = [switch_indices[end-4]+1, switch_indices[end]]
            rate_avg = Vector{BigFloat}(undef,3)
            for n in 1:3
                rate_avg[n] = trapz(t_array[idx[1]:idx[2]], r_inst[idx[1]:idx[2],n]) / (t_array[idx[2]] - t_array[idx[1]])
            end
            
            # define the conditions to reach steady state
            cond1 = abs(rate_avg[1] - rate_avg[2]) / max(1,abs(0.5*(rate_avg[1] + rate_avg[2])));
            cond2 = abs(rate_avg[1] - rate_avg[3]) / max(1,abs(0.5*(rate_avg[1] + rate_avg[3])));
            cond3 = abs(rate_avg[3] - rate_avg[2]) / max(1,abs(0.5*(rate_avg[3] + rate_avg[2])));

            if cond1<=tols[3] && cond2<=tols[3] && cond3<=tols[3]
                println("Integration terminated as steady state was achieved.")
                return true # steady state has been reached --> terminate integration
            else 
                # check that all rates are above 1e-4
                if any(abs.(rate_avg) .> 1e-4) == true
                    return false  # a given rate is greater than 1e-4 --> DON'T terminate integration
                else 
                    println("Integration terminated due to low rates.")
                    return true # no loop turnover frequency --> terminate integration
                end
            end
        else 
            return false
        end
    end

    function affect_steady_state!(integrator)
        terminate!(integrator)
    end
    
    # Combine 
    cb_S2 = DiscreteCallback(condition_S2, affect_S2!);
    cb_S1 = DiscreteCallback(condition_S1, affect_S1!);
    cb_steady_state = ContinuousCallback(condition_steady_state, affect_steady_state!);
    
    cbs = CallbackSet(cb_S2, cb_S1, cb_steady_state);
    
    return switch_times, cbs
end


function calc_rates(t,x_inst,switch_indices,k)
# Calculates instantaneous elementary rates & vector of catalyst state

    # Preallocate arrays
    len = length(x_inst[:,1])
    k_inst = Matrix{BigFloat}(undef,len,6)
    state_vector = Vector{Int64}(undef,len)
    
    # Populate a matrix of k(t), catalyst state vector
    for n in range(1,length(switch_indices)-2,step=2)
        # State 1 
        if n > 1
            k_inst[switch_indices[n]+1:switch_indices[n+1],:] .= k[:,1]'
            state_vector[switch_indices[n]+1:switch_indices[n+1]] .= 1
        else
            k_inst[switch_indices[n]:switch_indices[n+1],:] .= k[:,1]'
            state_vector[switch_indices[n]:switch_indices[n+1]] .= 1
        end
        # State 2
        k_inst[switch_indices[n+1]+1:switch_indices[n+2],:] .= k[:,2]'
        state_vector[switch_indices[n+1]+1:switch_indices[n+2]] .= 2
    end
    if switch_indices[end] < len
        # Simulation ends at state 1 (find #undef > populate with state 1 params)
        k_inst[switch_indices[end]+1:len,:] .= k[:,1]'
        state_vector[switch_indices[end]+1:len] .= 1
    end

    elem_rates_inst = Matrix{BigFloat}(undef,len,3)
    # Calculate elementary rate [1/s]
    elem_rates_inst[:,1] = k_inst[:,1].*x_inst[:,1] .- k_inst[:,4].*x_inst[:,2]
    elem_rates_inst[:,2] = k_inst[:,2].*x_inst[:,2] .- k_inst[:,5].*x_inst[:,3]
    elem_rates_inst[:,3] = k_inst[:,3].*x_inst[:,3] .- k_inst[:,6].*x_inst[:,1]

    return elem_rates_inst, state_vector, k_inst
end


## Find Indices of State Switches
function find_switches(switch_times,t)
# Maps state switch times to sol.t ODE solver output indices
    # Preallocate 
    switch_indices = Vector{Int64}(undef,length(switch_times)+1)
    switch_indices[1] = 1
    for n in range(1,length(switch_times))
        switch_indices[n+1] = findfirst(x->x in (switch_times[n]),t)
    end
    return switch_indices
end

function cutoff(t, switch_times)
    # find the time that the Simulation terminated
    t_cutoff=t[end];
    println(t_cutoff);
    switches = [];
    
    for i in range(1,length(switch_times))
        if switch_times[i] > t_cutoff
            switches = switch_times[1:i-1]
            break
        else 
            switches = switch_times[1:i]
        end
    end
    return t_cutoff, switches
end

function write_to_csv(fpath, df)
    if isfile(fpath) == false
       # If the file does not exst, write the file including column names
       col_names = ["Batch ID","Simulation ID", "alpha a", "alpha b", "alpha c", "beta a", "beta b", "beta c", "gamma B-A", "gamma C-A", "delta B-A", "delta C-A", "BEa", "frequency [1/s]", "ΔBEa [eV]", "Loop TOF [1/s]","Steady State Conditon"]; # 17 columns
        CSV.write(fpath,df, header=col_names)
    else    
        CSV.write(fpath, df, append=true)
    end
end

function define_loopTOF(rate_avg)
    #preallocate arrays
    rate_check = Vector{Float64}(undef,3)
    
    # compute the percent difference in the time averaged rates [if the rates are less than 1 use the absolute difference]
    rate_check[1] = abs(rate_avg[1] - rate_avg[2]) / max(1,(0.5*abs(rate_avg[1] + rate_avg[2]))) # check 1 and 2
    rate_check[2] = abs(rate_avg[1] - rate_avg[3]) / max(1,(0.5*abs(rate_avg[1] + rate_avg[3]))) # check 1 and 3
    rate_check[3] = abs(rate_avg[3] - rate_avg[2]) / max(1,(0.5*abs(rate_avg[3] + rate_avg[2]))) # check 3 and 2
    
    if any(rate_check .> 1e-3) == false
        # Steady-State is met with a loop TOF
        if any(abs.(rate_avg) .> 1e-4) == true # rates are greater than 1e-4
            loopTOF = (rate_avg[1]+rate_avg[2]+rate_avg[3])/3;
        else # rates are less than 1e-4
            loopTOF = 0; 
        end
    elseif any(abs.(rate_avg) .> 1e-4) == true
        # steady state isn't reached and some rates are still greater than 1e-4
        loopTOF = nothing;
    else 
        # steady state isn't reached but the rates are low
        loopTOF = 0;
    end
    println("Loop TOF: ", loopTOF)
    return loopTOF
end
   
   
   