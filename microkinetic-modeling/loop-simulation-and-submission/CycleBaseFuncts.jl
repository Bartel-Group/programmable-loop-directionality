#=
CycleBaseFuncts.jl

File contents:

Subfunctions necessary to run the dynamic cycle simulation. Contains the base functions to construct the reaction parameters, species balances, and rate equations needed to define the reaction. 

Functions included: 
    RxnParameters: Unpack the chemistry parameters for a given catalyst state and calculate the binding energies of each surface species and the activation energy of the forward direction of each elementary step. Then calculates the forward rate constant and the equilibrium constant, allowing for back calculation of the reverse rate constant.
--> Returns: k (concatonated array with the forward and reverse rate constants), K (array of Eq. constants)
    
    SpeciesBalDyn: Includes the species balances and rate equations for the dynamic process

Written in Julia 1.9

Author: Madeline Murphy
=#

function RxnParameters(BEa, gamma, delta, alpha, beta)
        # Constants
        T = 298.15                # temperature [K]
        kB = 8.61733034e-5        # Boltzmann constant [eV/K]
        h = 4.1357e-15            # Planck constant [eV-s]
        ev = 1.602e-19            # Converts Joules to Ev
         
        # Unpacking Parameters
        gamma_ab = gamma[1]; gamma_ac = gamma[2]
        delta_ab = delta[1]; delta_ac = delta[2]
        alpha_a = alpha[1]; alpha_b = alpha[2]; alpha_c = alpha[3]
        beta_a = beta[1]; beta_b = beta[2]; beta_c = beta[3]
        
        ## Binding Energies
        BEa = BEa
        BEb = gamma_ab*BEa + (1-gamma_ab)*delta_ab
        BEc = gamma_ac*BEa + (1-gamma_ac)*delta_ac
        
        ## Assuming negligible entropy change on the surface ΔH = ΔG. Same difference used for BEP relation and calculation of the Equilibrium constant
        deltaG = zeros(3,1)
        deltaG[1] = -BEb + BEa
        deltaG[2] = -BEc + BEb
        deltaG[3] = -BEa + BEc
        
        ## Activation Energies
        Ea = zeros(3,1)
        Ea[1] = alpha_a*deltaG[1] + beta_a
        Ea[2] = alpha_b*deltaG[2] + beta_b
        Ea[3] = alpha_c*deltaG[3] + beta_c  
        
        # if the parameters return a negative activation energy, exit the simulation
        # DOESN'T RUN OR RETURN ANYTHING
        #if any(Ea) < 0
            #exit()
        #end

        # Forward rate constant 
        kf = (kB*T/h)*exp.(-Ea/(kB*T))
        
        # Equilibrium constant
        K = exp.(-deltaG/(kB*T))
        
        # Reverse rate constant
        kr = kf./K
        
        k = vcat(kf,kr)
        return k, K
end


function SpeciesBal(dx,x,k,t)
    ## See Julia documentation
    # unpack parameter inputs
        # Rate equations
        r = [(k[1,1]*x[1]-k[4,1]*x[2]);
            (k[2,1]*x[2]-k[5,1]*x[3]);
            (k[3,1]*x[3]-k[6,1]*x[1])]
        
        # dx[1] = x[1] + x[2] + x[3] - 1.0
        dx[1] = r[3] - r[1]
        dx[2] = r[1] - r[2]
        dx[3] = r[2] - r[3]
    
        return nothing
end


function Directionality(k)
    τ = 1.0 ./k;
    
    J = Vector{Float64}(undef,3)
    J[1] = log(((τ[1,1] + τ[2,1])/(τ[4,1]+τ[5,1]))*((τ[1,2] + τ[2,2])/(τ[4,2]+τ[5,2])));
    J[2] = log(((τ[2,1] + τ[3,1])/(τ[5,1]+τ[6,1]))*((τ[2,2] + τ[3,2])/(τ[5,2]+τ[6,2])));
    J[3] = log(((τ[3,1] + τ[1,1])/(τ[6,1]+τ[4,1]))*((τ[3,2] + τ[1,2])/(τ[6,2]+τ[4,2])));
    
    J_avg = (J[1]+J[2]+J[3])/3;
    
    return J_avg
end 