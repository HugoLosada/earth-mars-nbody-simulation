import numpy as np
import matplotlib.pyplot as plt

from bodies import Body
from physics import (
    G,
    gravitational_acceleration,
    total_energy,
)
from integrators import velocity_verlet_step


def run_barycentric_simulation(number_of_steps):
    """
    Simulate one complete circular Sun-Earth orbit around
    the system barycenter.

    Parameters
    ----------
    number_of_steps : int
        Number of Velocity Verlet steps used during one orbit.

    Returns
    -------
    tuple
        Time step, maximum relative energy error,
        final relative-position error and radial error.
    """

    # ---------------------------------------------------------
    # 1. Physical parameters
    # ---------------------------------------------------------

    sun_mass = 1.0
    earth_mass = 3.004e-6
    separation = 1.0  # AU

    total_mass = sun_mass + earth_mass

    # ---------------------------------------------------------
    # 2. Distances from each body to the barycenter
    # ---------------------------------------------------------

    sun_radius = (
        earth_mass / total_mass
    ) * separation

    earth_radius = (
        sun_mass / total_mass
    ) * separation

    # ---------------------------------------------------------
    # 3. Angular velocity and orbital period
    # ---------------------------------------------------------

    angular_speed = np.sqrt(
        G * total_mass / separation**3
    )

    orbital_period = (
        2 * np.pi / angular_speed
    )

    # Choose dt so that:
    #
    # number_of_steps * dt = exactly one orbital period
    #
    dt = orbital_period / number_of_steps

    # ---------------------------------------------------------
    # 4. Create the Sun
    # ---------------------------------------------------------

    sun = Body(
        name="Sun",
        mass=sun_mass,
        position=(-sun_radius, 0.0),
        velocity=(
            0.0,
            -angular_speed * sun_radius,
        ),
    )

    # ---------------------------------------------------------
    # 5. Create the Earth
    # ---------------------------------------------------------

    earth = Body(
        name="Earth",
        mass=earth_mass,
        position=(earth_radius, 0.0),
        velocity=(
            0.0,
            angular_speed * earth_radius,
        ),
    )

    bodies = [sun, earth]

    # ---------------------------------------------------------
    # 6. Save initial quantities
    # ---------------------------------------------------------

    initial_relative_position = (
        earth.position - sun.position
    ).copy()

    initial_energy = total_energy(bodies)

    energy_errors = []

    # ---------------------------------------------------------
    # 7. Run one complete orbit
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 8. Calculate final errors
    # ---------------------------------------------------------

    final_relative_position = (
        earth.position - sun.position
    )

    final_distance = np.linalg.norm(
        final_relative_position
    )

    position_error = np.linalg.norm(
        final_relative_position
        - initial_relative_position
    )

    radial_error = abs(
        final_distance - separation
    )

    max_energy_error = max(
        energy_errors
    )

    return (
        dt,
        max_energy_error,
        position_error,
        radial_error,
    )


# -------------------------------------------------------------
# 9. Resolutions to compare
# -------------------------------------------------------------

step_counts = [
    100,
    200,
    500,
    1000,
    2000,
]


dt_values = []
energy_errors = []
position_errors = []
radial_errors = []


# -------------------------------------------------------------
# 10. Run the convergence study
# -------------------------------------------------------------

print(
    f"{'Steps':>10} "
    f"{'dt':>15} "
    f"{'Max energy error':>20} "
    f"{'Position error':>20} "
    f"{'Radial error':>20}"
)


for number_of_steps in step_counts:

    (
        dt,
        max_energy_error,
        position_error,
        radial_error,
    ) = run_barycentric_simulation(
        number_of_steps
    )

    dt_values.append(dt)
    energy_errors.append(
        max_energy_error
    )
    position_errors.append(
        position_error
    )
    radial_errors.append(
        radial_error
    )

    print(
        f"{number_of_steps:10d} "
        f"{dt:15.6e} "
        f"{max_energy_error:20.6e} "
        f"{position_error:20.6e} "
        f"{radial_error:20.6e}"
    )


# -------------------------------------------------------------
# 11. Second-order reference curve
# -------------------------------------------------------------

dt_array = np.array(
    dt_values
)

position_error_array = np.array(
    position_errors
)

second_order_reference = (
    position_error_array[0]
    * (
        dt_array / dt_array[0]
    )**2
)



for i in range(len(dt_values) - 1):
    p = np.log(
        position_errors[i + 1] / position_errors[i]
    ) / np.log(
        dt_values[i + 1] / dt_values[i]
    )

    print(
        f"{dt_values[i]:.6e} -> "
        f"{dt_values[i + 1]:.6e}: "
        f"p = {p:.6f}"
    )
    

# -------------------------------------------------------------
# 12. Plot convergence
# -------------------------------------------------------------

plt.loglog(
    dt_values,
    position_errors,
    marker="o",
    label="Position error",
)

plt.loglog(
    dt_values,
    second_order_reference,
    linestyle="--",
    label="Second-order reference",
)

plt.xlabel(
    "Time step dt [years]"
)

plt.ylabel(
    "Position error [AU]"
)

plt.title(
    "Barycentric Velocity Verlet convergence"
)

plt.grid(
    True,
    which="both",
)

plt.legend()

plt.show()


print("\nObserved convergence order:")

