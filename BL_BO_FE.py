import numpy as np
from scipy.integrate import solve_ivp
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.sampler import Lhs, Sobol
import math
import threading

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Button,
    Select,
    TextInput,
    Div,
    Spinner,
    Paragraph,
)

# --- 1. Define Model Parameters (Constants of the Lutein system) ---
# These constants describe the biological and physical properties of the system.
U_M = 0.152  # Maximum specific growth rate (1/h)
U_D = 5.95e-3  # Specific death/decay rate (1/h)
K_N = 30.0e-3  # Monod constant for nitrate (g/L)
Y_NX = 0.305  # Yield coefficient of biomass on nitrate (g/g)
K_M = 0.350e-3 * 2  # Maximum specific lutein production rate (g/g-h)
K_D = 3.71 * 0.05 / 90  # Specific lutein degradation rate (L/g-h)
K_NL = 10.0e-3  # Monod constant for nitrate for lutein production (g/L)
K_S = 142.8  # Light saturation constant for growth (umol/m2-s)
K_I = 214.2  # Light inhibition constant for growth (umol/m2-s)
K_SL = 320.6  # Light saturation constant for lutein production (umol/m2-s)
K_IL = 480.9  # Light inhibition constant for lutein production (umol/m2-s)
TAU = 0.120  # Light attenuation coefficient (m2/g)
KA = 0.0  # Placeholder constant

# --- Global Variables for ODE solver ---
# These are used by the 'pbr' function. They are set globally because
# solve_ivp's standard interface doesn't easily pass extra parameters
# without wrapping the function in a lambda or class. This approach is kept
# from the original code for consistency.
C_x0_model = 0.5
C_N0_model = 1.0
F_in_model = 8e-3
C_N_in_model = 10.0
I0_model = 150.0


# --- 2. Define the Photobioreactor ODE Model ---
def pbr(t, C):
    """
    Defines the system of Ordinary Differential Equations for the photobioreactor.
    C[0]: Biomass concentration (C_X)
    C[1]: Nitrate concentration (C_N)
    C[2]: Lutein concentration (C_L)
    """
    C_X, C_N, C_L = C

    # Avoid division by zero or negative concentrations in the model
    if C_X < 1e-9: C_X = 1e-9
    if C_N < 1e-9: C_N = 1e-9
    if C_L < 1e-9: C_L = 1e-9

    # Light availability calculation (Beer-Lambert law)
    I = 2 * I0_model * (np.exp(-(TAU * 0.01 * 1000 * C_X)))

    # Light scaling factors for growth (u) and lutein production (k)
    Iscaling_u = I / (I + K_S + I ** 2 / K_I)
    Iscaling_k = I / (I + K_SL + I ** 2 / K_IL)

    # Specific rates for growth and lutein production
    u0 = U_M * Iscaling_u
    k0 = K_M * Iscaling_k

    # ODEs for each component
    dCxdt = u0 * C_N * C_X / (C_N + K_N) - U_D * C_X
    dCndt = -Y_NX * u0 * C_N * C_X / (C_N + K_N) + F_in_model * C_N_in_model
    dCldt = k0 * C_N * C_X / (C_N + K_NL) - K_D * C_L * C_X

    return np.array([dCxdt, dCndt, dCldt])


# --- 3. Helper function to evaluate the model and objective ---
def _evaluate_lutein_model_objective(C_x0, C_N0, F_in, C_N_in, I0):
    """
    Sets up and runs a single simulation to find the final lutein concentration.
    This is the core function that the optimizer will try to minimize (negative of).
    """
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = (
        C_x0, C_N0, F_in, C_N_in, I0
    )

    time_span = [0, 150]
    initial_conditions = np.array([C_x0_model, C_N0_model, 0.0])

    # Solve the ODE system
    sol = solve_ivp(
        pbr, time_span, initial_conditions, t_eval=[150], method="RK45"
    )
    final_lutein_concentration = sol.y[2, -1]

    # The optimizer minimizes, so we return the negative of our goal (lutein)
    # Return a large number if the result is invalid to guide the search away.
    if final_lutein_concentration <= 0 or not np.isfinite(final_lutein_concentration):
        return 1e6  # A large penalty for non-physical results

    return -final_lutein_concentration


# --- 4. Define the Search Space and Objective Function for skopt ---
dimensions = [
    Real(0.2, 2.0, name="C_x0"),
    Real(0.2, 2.0, name="C_N0"),
    Real(1e-3, 1.5e-2, name="F_in"),
    Real(5.0, 15.0, name="C_N_in"),
    Real(100.0, 200.0, name="I0"),
]

@use_named_args(dimensions)
def objective_function(C_x0, C_N0, F_in, C_N_in, I0):
    """Wrapper for skopt to call the evaluation function with named arguments."""
    return _evaluate_lutein_model_objective(C_x0, C_N0, F_in, C_N_in, I0)


# --- 5. Bokeh Application Setup ---

# --- Data Sources ---
# Source for the convergence plot (iteration vs. best lutein found)
convergence_source = ColumnDataSource(data=dict(iter=[], best_lutein=[]))

# Source for the final simulation plot (time vs. concentrations)
simulation_source = ColumnDataSource(
    data=dict(time=[], C_X=[], C_N=[], C_L=[])
)


# --- Callbacks and Update Functions ---

def run_optimization():
    """
    Triggered by the 'Start' button.
    Reads UI parameters, disables widgets, and starts the optimization in a new thread.
    """
    # Disable controls during run
    start_button.disabled = True
    for widget in control_widgets:
        widget.disabled = True

    # Clear previous results and plots
    status_div.text = "🔄 Optimization in progress... please wait."
    results_div.text = ""
    convergence_source.data = dict(iter=[], best_lutein=[])
    simulation_source.data = dict(time=[], C_X=[], C_N=[], C_L=[])

    # Start the potentially long-running optimization in a separate thread
    # This prevents the UI from freezing.
    thread = threading.Thread(target=threaded_optimization_worker)
    thread.start()


def threaded_optimization_worker():
    """
    The main worker function that runs gp_minimize.
    This function runs in a separate thread to avoid blocking the Bokeh server.
    """
    # Get parameters from the UI widgets
    n_calls = int(n_calls_input.value)
    n_initial = int(n_initial_input.value)
    surrogate_model = surrogate_select.value
    acq_func = acq_func_select.value
    sampler_choice = sampler_select.value

    # Validate inputs
    if n_initial >= n_calls:
        doc.add_next_tick_callback(
            lambda: update_status(
                "❌ Error: Number of initial points must be less than total iterations.",
                is_error=True,
            )
        )
        return

    # --- Generate Initial Points ---
    x0, y0 = None, None
    if n_initial > 0:
        if sampler_choice == 'lhs':
            sampler = Lhs(lhs_type="centered", criterion="maximin")
            x0 = sampler.generate(dimensions, n_samples=n_initial, random_state=42)
        elif sampler_choice == 'sobol':
            sampler = Sobol()
            x0 = sampler.generate(dimensions, n_samples=n_initial, random_state=42)
        # 'random' is handled by gp_minimize directly if x0 is None

    try:
        # Define the callback for gp_minimize to update the plot live
        def skopt_callback(res):
            iteration = len(res.func_vals)
            best_lutein = -res.fun
            # Schedule UI update to be run by the Bokeh server's main IOLoop
            doc.add_next_tick_callback(
                lambda: update_convergence_plot(iteration, best_lutein)
            )

        # --- Run Bayesian Optimization ---
        result = gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            base_estimator=surrogate_model,
            acq_func=acq_func,
            n_calls=n_calls,
            x0=x0,
            n_initial_points=n_initial if x0 is None else 0,
            random_state=42,
            callback=[skopt_callback],
        )

        # --- Process and Display Final Results ---
        # Schedule the final updates to be run by the main IOLoop
        doc.add_next_tick_callback(
            lambda: process_final_results(result)
        )

    except Exception as e:
        error_message = f"❌ An error occurred during optimization: {e}"
        doc.add_next_tick_callback(lambda: update_status(error_message, is_error=True))


def process_final_results(result):
    """
    Once optimization is complete, this function updates the results text
    and runs/plots the final simulation with the optimal parameters.
    """
    status_div.text = "✅ Optimization complete."

    # --- Display Optimal Parameters ---
    max_lutein = -result.fun
    optimal_params = {dim.name: val for dim, val in zip(dimensions, result.x)}

    results_html = f"<h3>Optimal Results</h3>"
    results_html += f"<b>Maximum Lutein Concentration:</b> {-result.fun:.4f} g/L<br/>"
    results_html += "<b>Optimal Parameters:</b><ul>"
    for param, value in optimal_params.items():
        results_html += f"<li><b>{param}:</b> {value:.4f}</li>"
    results_html += "</ul>"
    results_div.text = results_html

    # --- Run and Plot the Final Simulation ---
    C_x0, C_N0, F_in, C_N_in, I0 = result.x
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = (
        C_x0, C_N0, F_in, C_N_in, I0
    )
    time_span = [0, 150]
    t_eval = np.linspace(time_span[0], time_span[1], 300)
    initial_conditions = np.array([C_x0_model, C_N0_model, 0.0])

    sol = solve_ivp(
        pbr, time_span, initial_conditions, t_eval=t_eval, method="RK45"
    )

    # Update the simulation plot's data source
    simulation_source.data = {
        "time": sol.t,
        "C_X": sol.y[0],
        "C_N": sol.y[1],
        "C_L": sol.y[2],
    }

    # Re-enable controls
    start_button.disabled = False
    for widget in control_widgets:
        widget.disabled = False


def update_convergence_plot(iteration, best_lutein):
    """Updates the convergence plot data source by streaming new data."""
    convergence_source.stream({"iter": [iteration], "best_lutein": [best_lutein]})

def update_status(message, is_error=False):
    """Updates the status Div and re-enables controls in case of an error."""
    status_div.text = message
    if is_error:
        start_button.disabled = False
        for widget in control_widgets:
            widget.disabled = False


# --- UI Widgets ---
doc = curdoc()
doc.title = "Lutein Production Optimizer"

# --- Control Widgets ---
title_div = Div(text="<h1>Lutein Production Bayesian Optimizer</h1>")
description_p = Paragraph(
    text="""
    This application uses Bayesian Optimization to find the optimal operating conditions
    for a photobioreactor to maximize lutein production. Adjust the optimizer settings
    below and click 'Start Optimization' to begin.
    """, width=400
)

surrogate_select = Select(
    title="Surrogate Model (Regressor):",
    value="GP",
    options=[
        ("GP", "Gaussian Process"),
        ("RF", "Random Forest"),
        ("ET", "Extra Trees"),
    ],
)
acq_func_select = Select(
    title="Acquisition Function:",
    value="gp_hedge",
    options=[
        ("gp_hedge", "GP Hedge (Automatic)"),
        ("EI", "Expected Improvement"),
        ("PI", "Probability of Improvement"),
        ("LCB", "Lower Confidence Bound"),
    ],
)
sampler_select = Select(
    title="Initial Sampling Method:",
    value="lhs",
    options=[
        ("lhs", "Latin Hypercube (LHS)"),
        ("sobol", "Sobol Sequence"),
        ("random", "Random Sampling"),
    ],
)
n_calls_input = Spinner(
    title="Total Optimization Iterations:",
    low=2, step=1, value=50, width=150
)
n_initial_input = Spinner(
    title="Number of Initial Points:",
    low=1, step=1, value=10, width=150
)
start_button = Button(label="Start Optimization", button_type="success", width=400)
start_button.on_click(run_optimization)

status_div = Div(text="🟢 Ready to start.")
results_div = Div(text="")

control_widgets = [
    surrogate_select,
    acq_func_select,
    sampler_select,
    n_calls_input,
    n_initial_input,
]

controls = column(
    title_div,
    description_p,
    surrogate_select,
    acq_func_select,
    sampler_select,
    row(n_calls_input, n_initial_input),
    start_button,
    status_div,
    results_div,
    width=420,
)

# --- Plots ---
# Convergence Plot
p_conv = figure(
    height=400,
    width=600,
    title="Optimization Convergence",
    x_axis_label="Iteration",
    y_axis_label="Max Lutein Found (g/L)",
)
p_conv.line(
    x="iter", y="best_lutein", source=convergence_source, line_width=2,
    legend_label="Best Objective"
)
p_conv.legend.location = "bottom_right"

# Final Simulation Plot
p_sim = figure(
    height=400,
    width=600,
    title="Simulation with Optimal Parameters",
    x_axis_label="Time (hours)",
    y_axis_label="Concentration (g/L)",
)
p_sim.line(x="time", y="C_X", source=simulation_source, color="green", line_width=2, legend_label="Biomass (C_X)")
p_sim.line(x="time", y="C_N", source=simulation_source, color="blue", line_width=2, legend_label="Nitrate (C_N)")
p_sim.line(x="time", y="C_L", source=simulation_source, color="orange", line_width=3, legend_label="Lutein (C_L)")
p_sim.legend.location = "top_left"
p_sim.legend.click_policy = "hide" # Allows toggling lines on/off


# --- Layout ---
plots = column(p_conv, p_sim)
layout = row(controls, plots)
doc.add_root(layout)
