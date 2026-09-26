import numpy as np
import matplotlib.pyplot as plt
from bodies import Body
from physics import (
    circular_orbit_speed,
    gravitational_acceleration,
    total_energy,
)
from integrators import velocity_verlet_step


def run_simulation(dt):
    """
    Run a one-year Earth-Sun simulation for a given time step.

    Parameters
    ----------
    dt : float
        Integration time step in years.

    Returns
    -------
    tuple
        Maximum relative energy error,
        final position error,
        and final radial error.
    """

    # Physical parameters
    earth_radius = 1.0
    simulation_time = 1.0

    # Calculate Earth's circular orbital speed
    earth_speed = circular_orbit_speed(
        central_mass=1.0,
        radius=earth_radius,
    )

    # Create a new Sun for this simulation
    sun = Body(
        name="Sun",
        mass=1.0,
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
    )

    # Create a new Earth for this simulation
    earth = Body(
        name="Earth",
        mass=3.004e-6,
        position=(earth_radius, 0.0),
        velocity=(0.0, earth_speed),
    )

    bodies = [sun, earth]

    # Number of integration steps needed
    number_of_steps = int(
        simulation_time / dt
    )

    # Initial quantities for comparison
    initial_relative_position = (
        earth.position - sun.position
    ).copy()

    initial_energy = total_energy(bodies)

    energy_errors = []

    # Run the numerical simulation
    for step in range(number_of_steps):

        velocity_verlet_step(
            bodies,
            dt,
            gravitational_acceleration,
        )

        current_energy = total_energy(bodies)

        relative_energy_error = abs(
            (current_energy - initial_energy)
            / initial_energy
        )

        energy_errors.append(
            relative_energy_error
        )

    # Final relative position
    final_relative_position = (
        earth.position - sun.position
    )

    # Final Earth-Sun distance
    final_distance = np.linalg.norm(
        final_relative_position
    )

    # Error in orbital radius
    radial_error = abs(
        final_distance - earth_radius
    )

    # Error in position after one year
    position_error = np.linalg.norm(
        final_relative_position
        - initial_relative_position
    )

    # Largest energy deviation during the simulation
    max_energy_error = max(
        energy_errors
    )

    return (
        max_energy_error,
        position_error,
        radial_error,
    )


# Time steps that we want to compare
dt_values = [
    0.01,
    0.005,
    0.002,
    0.001,
    0.0005,
]


print(
    f"{'dt':>10} "
    f"{'Max energy error':>20} "
    f"{'Position error':>20} "
    f"{'Radial error':>20}"
)

position_errors = []
energy_errors = []
radial_errors = []

for dt in dt_values:

    max_energy_error, position_error, radial_error = (
        run_simulation(dt)
    )
    energy_errors.append(max_energy_error)
    position_errors.append(position_error)
    radial_errors.append(radial_error)

    print(
        f"{dt:10.4g} "
        f"{max_energy_error:20.6e} "
        f"{position_error:20.6e} "
        f"{radial_error:20.6e}"
    )


reference_error = (
    position_errors[0]
    * (
        np.array(dt_values)
        / dt_values[0]
    )**2
)
plt.loglog(
    dt_values,
    reference_error,
    linestyle="--",
    label="Second-order reference",
)
plt.loglog(
    dt_values,
    position_errors,
    marker="o",
    label="Position error",
)

plt.xlabel("Time step dt [years]")
plt.ylabel("Position error [AU]")

plt.title(
    "Velocity Verlet convergence"
)

plt.grid(
    True,
    which="both",
)

plt.legend()

plt.show()