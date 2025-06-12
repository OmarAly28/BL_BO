import numpy as np
from scipy.integrate import solve_ivp
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.sampler import Lhs, Sobol
import math

# --- Configuration Constants for User Experience ---
# Rough estimate of time per iteration for warning message (in seconds).
# Based on reported 8 mins for 200 iterations -> 480/200 = 2.4 sec/iter.
EST_SEC_PER_ITER = 2.5
# Threshold above which a warning is issued
WARN_THRESHOLD_ITERATIONS = 100
# Absolute maximum recommended iterations for CLI
MAX_RECOMMENDED_ITERATIONS = 300

# --- 1. Define Model Parameters (Constants of the Lutein system) ---
U_M = 0.152 # Maximum specific growth rate (1/h)
U_D = 5.95*1e-3 # Specific death/decay rate (1/h)
K_N = 30.0*1e-3 # Monod constant for nitrate (g/L)
Y_NX = 0.305 # Yield coefficient of biomass on nitrate (g/g)
K_M = 0.350*1e-3*2 # Maximum specific lutein production rate (g/g-h)
K_D = 3.71*0.05/90 # Specific lutein degradation rate (L/g-h)
K_NL = 10.0*1e-3 # Monod constant for nitrate for lutein production (g/L)
K_S = 142.8 # Light saturation constant for growth (umol/m2-s)
K_I = 214.2 # Light inhibition constant for growth (umol/m2-s)
K_SL = 320.6 # Light saturation constant for lutein production (umol/m2-s)
K_IL = 480.9 # Light inhibition constant for lutein production (umol/m2-s)
TAU = 0.120 # Light attenuation coefficient (m2/g)
KA = 0.0 # Placeholder constant (not used in current ODEs)

# --- Global Variables for pbr function (these will be updated by objective_function or _evaluate_lutein_model_objective) ---
C_x0_model = 0.5 # Default initial biomass concentration
C_N0_model = 1.0 # Default initial nitrate concentration
F_in_model = 8e-3 # Default nitrate inflow rate
C_N_in_model = 10.0 # Default inlet nitrate concentration
I0_model = 150.0 # Default incident light intensity

# --- 2. Define the Photobioreactor ODE Model (pbr function) ---
def pbr(t, C):
    C_X = C[0]
    C_N = C[1]
    C_L = C[2]

    if C_X < 1e-9: C_X = 1e-9
    if C_N < 1e-9: C_N = 1e-9
    if C_L < 1e-9: C_L = 1e-9
    
    I = 2 * I0_model * (np.exp(-(TAU * 0.01 * 1000 * C_X)))
    Iscaling_u = I / (I + K_S + I**2 / K_I)
    Iscaling_k = I / (I + K_SL + I**2 / K_IL)
    u0 = U_M * Iscaling_u
    k0 = K_M * Iscaling_k

    dCxdt = u0 * C_N * C_X / (C_N + K_N) - U_D * C_X
    dCndt = -Y_NX * u0 * C_N * C_X / (C_N + K_N) + F_in_model * C_N_in_model
    dCldt = k0 * C_N * C_X / (C_N + K_NL) - K_D * C_L * C_X
    
    return np.array([dCxdt, dCndt, dCldt])

# --- Helper function for evaluating the Lutein model and objective ---
def _evaluate_lutein_model_objective(C_x0, C_N0, F_in, C_N_in, I0):
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model = C_x0
    C_N0_model = C_N0
    F_in_model = F_in
    C_N_in_model = C_N_in
    I0_model = I0

    time_span = [0, 150]
    initial_conditions = np.array([C_x0_model, C_N0_model, 0.0])

    sol = solve_ivp(pbr, time_span, initial_conditions, t_eval=[150])
    final_lutein_concentration = sol.y[2, -1]
    
    if final_lutein_concentration <= 0:
        return np.inf

    return -final_lutein_concentration

# --- 3. Define the Objective Function for Bayesian Optimization (for skopt) ---
@use_named_args([
    Real(0.2, 2.0, name='C_x0'),
    Real(0.2, 2.0, name='C_N0'),
    Real(1e-3, 1.5e-2, name='F_in'),
    Real(5.0, 15.0, name='C_N_in'),
    Real(100.0, 200.0, name='I0')
])
def objective_function(C_x0, C_N0, F_in, C_N_in, I0):
    return _evaluate_lutein_model_objective(C_x0, C_N0, F_in, C_N_in, I0)

# --- 4. Define the Search Space for Optimization ---
dimensions = [
    Real(0.2, 2.0, name='C_x0'),
    Real(0.2, 2.0, name='C_N0'),
    Real(1e-3, 1.5e-2, name='F_in'),
    Real(5.0, 15.0, name='C_N_in'),
    Real(100.0, 200.0, name='I0')
]

# --- CLI User Input Helper Functions ---
def get_user_choice(prompt, options):
    """
    Prompts the user to choose from a list of numbered options and validates the input.
    """
    choice = -1
    while choice not in range(1, len(options) + 1):
        print(prompt)
        for i, option in enumerate(options):
            print(f"{i+1}. {option['label']}")
        try:
            choice = int(input("Enter the number of your choice: "))
            if choice not in range(1, len(options) + 1):
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return options[choice - 1]['value']

def get_integer_input(prompt, min_val=1, max_val=None, default_val=None):
    """
    Prompts the user for an integer input within an optional range.
    """
    while True:
        full_prompt = prompt
        if default_val is not None:
            full_prompt += f" (default: {default_val})"
        
        input_str = input(full_prompt + ": ")

        if input_str == '' and default_val is not None:
            return default_val
        
        try:
            value = int(input_str)
            if value < min_val:
                print(f"Value must be at least {min_val}.")
            elif max_val is not None and value > max_val:
                print(f"Value cannot exceed {max_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

if __name__ == '__main__':
    print("\n--- Lutein Production Bayesian Optimization (CLI) ---")

    # --- Step 1: Get User Choice for Surrogate Model ---
    surrogate_model_options = [
        {'value': 'GP', 'label': 'Gaussian Process (default)'},
        {'value': 'RF', 'label': 'Random Forest'},
        {'value': 'ET', 'label': 'Extra Trees'}
    ]
    surrogate_model_choice = get_user_choice(
        "\nChoose Surrogate Model (Regressor):",
        surrogate_model_options
    )

    # --- Step 2: Get User Choice for Acquisition Function ---
    acquisition_function_options = [
        {'value': 'gp_hedge', 'label': 'GP Hedge (Balances EI, PI, LCB)'},
        {'value': 'EI', 'label': 'Expected Improvement'},
        {'value': 'PI', 'label': 'Probability of Improvement'},
        {'value': 'LCB', 'label': 'Lower Confidence Bound'}
    ]
    acquisition_function_choice = get_user_choice(
        "\nChoose Acquisition Function:",
        acquisition_function_options
    )

    # --- Step 3: Get User Input for Total Number of Iterations ---
    num_iterations = get_integer_input(
        f"Enter the total number of optimization iterations (e.g., 50-200, max {MAX_RECOMMENDED_ITERATIONS})",
        min_val=1, 
        max_val=MAX_RECOMMENDED_ITERATIONS, # Enforce maximum recommended iterations
        default_val=50
    )

    # --- Step 4: Get User Choice for Initial Sampling Method ---
    initial_sampler_options = [
        {'value': 'random', 'label': 'Random Sampling (Default)'},
        {'value': 'lhs', 'label': 'Latin Hypercube Sampling (LHS)'},
        {'value': 'sobol', 'label': 'Sobol Sequence'}
    ]
    initial_sampler_choice = get_user_choice(
        "\nChoose Initial Sampling Method:",
        initial_sampler_options
    )

    # --- Step 5: Get User Input for Number of Initial Points ---
    # Max initial points should be at least 1, and at most (total_iterations - 1)
    # to allow at least one model-guided step after initial points.
    max_initial_points_allowed = max(1, num_iterations - 1)
    # Ensure default is also within the allowed range
    default_initial_points = min(10, max_initial_points_allowed)

    num_initial_points = get_integer_input(
        f"Enter the number of initial points for sampling (min 1, max {max_initial_points_allowed})",
        min_val=1,
        max_val=max_initial_points_allowed,
        default_val=default_initial_points
    )

    # --- WARN THE USER ABOUT LONG RUNTIMES ---
    if num_iterations > WARN_THRESHOLD_ITERATIONS:
        estimated_time_seconds = num_iterations * EST_SEC_PER_ITER
        estimated_minutes = math.ceil(estimated_time_seconds / 60)
        print(f"\nWARNING: You have chosen {num_iterations} iterations.")
        print(f"  This optimization is estimated to take around {estimated_minutes} minutes to complete on your system.")
        print("  Please be patient. Consider reducing iterations for faster results if needed.")
        input("Press Enter to continue or Ctrl+C to abort...") # Pause execution for user acknowledgment

    # --- Generate Initial Points based on User Choice ---
    initial_points_x = []
    initial_points_y = []

    if num_initial_points > 0:
        print(f"\nGenerating {num_initial_points} initial points using {initial_sampler_choice} sampling...")
        # Seed numpy for reproducibility of custom random generation if not using skopt.sampler with random_state
        np.random.seed(42) 
        if initial_sampler_choice == 'random':
            for _ in range(num_initial_points):
                point = [dim.rvs()[0] for dim in dimensions] 
                initial_points_x.append(point)
        elif initial_sampler_choice == 'lhs':
            sampler = Lhs(lhs_type="centered", criterion="maximin")
            initial_points_x = sampler.generate(dimensions, n_samples=num_initial_points, random_state=42)
        elif initial_sampler_choice == 'sobol':
            sampler = Sobol()
            initial_points_x = sampler.generate(dimensions, n_samples=num_initial_points, random_state=42)
        
        # Evaluate each initial point using the *undecorated* helper function.
        for i, p_values_list in enumerate(initial_points_x):
            print(f"  Evaluating initial point {i+1}/{num_initial_points}...")
            initial_points_y.append(_evaluate_lutein_model_objective(*p_values_list))

    # --- Display chosen configuration ---
    print(f"\nRunning optimization with:")
    print(f"  Surrogate Model: {surrogate_model_choice}")
    print(f"  Acquisition Function: {acquisition_function_choice}")
    print(f"  Total Iterations (n_calls): {num_iterations}")
    print(f"  Initial Points Method: {initial_sampler_choice} (Number: {num_initial_points})")
    print("\nOptimization in progress (this may take a few moments)...")

    optimization_history = []
    def callback(res):
        current_best_lutein = -res.fun
        iteration_data = {
            'iteration': len(res.func_vals),
            'objective_value': -res.func_vals[-1],
            'best_objective_so_far': current_best_lutein
        }
        optimization_history.append(iteration_data)

    # --- 5. Run Bayesian Optimization with chosen parameters ---
    try:
        result = gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            base_estimator=surrogate_model_choice,
            acq_func=acquisition_function_choice,
            n_calls=num_iterations,
            x0=initial_points_x if num_initial_points > 0 else None,
            y0=initial_points_y if num_initial_points > 0 else None,
            n_initial_points=0 if num_initial_points > 0 else 10, # If x0/y0 are provided, set to 0. Otherwise, use skopt's default 10 random.
            random_state=42,
            callback=[callback]
        )

        # --- 6. Print Final Results ---
        print("\n--- Bayesian Optimization Results ---")
        ##print(f"Optimal (minimized) objective value: {result.fun:.4f}")
        print(f"Maximum Lutein concentration found: {-result.fun:.4f} g/L")

        print("\nOptimal parameters found:")
        optimal_params_dict = {name.name: value for name, value in zip(dimensions, result.x)}
        for param, value in optimal_params_dict.items():
            print(f"  {param}: {value:.4f}")

    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
