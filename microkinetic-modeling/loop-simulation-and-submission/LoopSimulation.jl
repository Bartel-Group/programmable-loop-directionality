#=
FILE FOR MSI-FRIENDLY Cyclic Dynamic JULIA FILE TO RUN IN BATCH SUBMISSION JOBS
File contents:

Function that runs a simulation of a dynamic cyclic reaction using a full set of parameters to describe the reaction coordinate of the 3 species cycle as well as the dynamic perturbation of the catalyst. This simulation operates via a square waveform shape for catalyst operatation where State 1 is described by the binding energy BEa and State 2 is defined by the binding energy BEa + ΔBEa

14 ARGS passed into function: 
    ARGS[1-3] = alpha --> BEP slope for species A*, B*, C*
    ARGS[4-6] = beta --> BEP offset for species A*, B*, C*
    ARGS[7-8] = gamma --> Linear scaling slope relating BEa to BEb, BEa to BEc
    ARGS[9-10] = delta --> Linear scaling offset. equal binding energy of BEa and BEb, BEa and BEc
    ARGS[11] = delBEa --> amplitude of catalyst oscillation
    ARGS[12] = Job ID
    ARGS[13] = Batch ID
    ARGS[14] = file path --> location for saving output data

Required packages: using DifferentialEquations, Trapz, DataFrames, Printf, CSV, NBInclude
Required files: CycleBaseFuncts.jl and Callbacks.jl
Written in Julia 1.9

Author: Madeline Murphy
=#

## Load packages and other .jl files needed 
using DifferentialEquations, Trapz, DataFrames, Printf, CSV, NBInclude
include("CycleBaseFuncts.jl")
include("Callbacks.jl")

setprecision(100)

## Unpack ARGS
## Bronsted-Evans Polanyi 
alpha = [parse(Float64, ARGS[1]); parse(Float64, ARGS[2]); parse(Float64, ARGS[3])];
beta = [parse(Float64, ARGS[4]); parse(Float64, ARGS[5]); parse(Float64, ARGS[6])]; # [=] eV

## Linear Scaling
gamma = [parse(Float64, ARGS[7]); parse(Float64, ARGS[8])];
delta = [parse(Float64, ARGS[9]); parse(Float64, ARGS[10])];  # [=] eV

## Binding descriptor 
BEa = 0.8;
delBEa = parse(Float64, ARGS[11]);
f = 50;
DC = 0.5;
tau = [DC/f; (1-DC)/f];
Numosc_max = 150;

SimID = ARGS[12];
BatchID = ARGS[13];
fpath = ARGS[14];

println("Batch number: ", BatchID, " Simulation number: ", SimID)

# set tolerances for integration and Steady State
tols = [1e-8, 1e-10, 1e-3];
u0 = [(1/3), (1/3), (1/3)];  # initial coverages

# Mass Matrix
M =[1 0 0;
    0 1 0;
    0 0 1];

# Define Kinetics Parameters
    # Catalyst State 1
    k1, K1 = RxnParameters(BEa, gamma, delta, alpha, beta);
    # Catalyst State 2
    k2, K2 = RxnParameters((BEa+delBEa), gamma, delta, alpha, beta);

    k = hcat(k1,k2);
    K = hcat(K1,K2);

    ## Builds microkinetic model for inputs specified above
    fun = ODEFunction(SpeciesBal, mass_matrix=M)
    mkm = ODEProblem(fun, big.(u0), big.([0.,Numosc_max*sum(tau)]), big.(k[:,1]));

    switch_times, cbs = build_callbacks(tau,Numosc_max,k)

    # Solve model
     sol = solve(mkm, Rosenbrock23(), callback=cbs, tstops=switch_times,
            reltol=tols[1], abstol=tols[2], dtmin=1e-40, dtmax=sum(tau)/100, maxiters=1e7, dense=false);

    # clear up memory 
     # Parse ODE solver outputs
     t = sol.t
     x_inst = Array(Array(sol)')

     # Cutoff the switch_times based on integration length and return time simulation terminated
     t_cutoff, switches = cutoff(t,switch_times)
     
     ## Analyze outputes
     switch_indices = find_switches(switches,t)

    # Calculate elementary rates and the catalyst state vector
    elem_rates_inst, state_vector = calc_rates(t,x_inst,switch_indices,k);
    
    # Calculate time-averaged quantities 
    idx = [switch_indices[end-4]+1, switch_indices[end]]
        
    rate_avg = Vector{Float64}(undef,3)
    for n in 1:3
             rate_avg[n] = trapz(t[idx[1]:idx[2]], elem_rates_inst[idx[1]:idx[2],n]) / (t[idx[2]] - t[idx[1]])
    end
  
    # define the loop turnover frequency
    loopTOF = define_loopTOF(rate_avg)

     if loopTOF == nothing
         ## Export the parameters and note to rerurn --> Steady State was NOT reached
         # Create DataFrame
         df = DataFrame(BatchID = BatchID, SimID = SimID, alpha_a = alpha[1], alpha_b = alpha[2], alpha_c = alpha[3], beta_a = beta[1], beta_b = beta[2], beta_c = beta[3], gamma_BA = gamma[1], gamma_CA = gamma[2], delta_BA = delta[1], delta_CA = delta[2], BEa = BEa*ones(1),frequency = f*ones(1), ΔBEa = delBEa*ones(1), Loop_TOF = "Not-defined", Steady_State = false)
    
         write_to_csv(fpath, df)
         println("Solutions has not reached steady-state ... exporting results and exiting")
    # employ garbage collection to clear up Memory 
        GC.gc()
         exit()
     else 
     # Steady State has been reached
        # Create DataFrame
        df = DataFrame(BatchID = BatchID, SimID = SimID, alpha_a = alpha[1], alpha_b = alpha[2], alpha_c = alpha[3], beta_a = beta[1], beta_b = beta[2], beta_c = beta[3], gamma_BA = gamma[1], gamma_CA = gamma[2], delta_BA = delta[1], delta_CA = delta[2], BEa = BEa*ones(1),frequency = f*ones(1), ΔBEa = delBEa*ones(1), Loop_TOF = loopTOF, Steady_State = true)
           
        write_to_csv(fpath, df)
    # employ garbage collection to clear up Memory 
        GC.gc()
     end

## End file with exit() so once it finishes, your MSI batch job will be killed instead of sitting idle. 
exit()