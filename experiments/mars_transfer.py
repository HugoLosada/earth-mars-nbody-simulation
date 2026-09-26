import numpy as np

from bodies import Body
from physics import (
    G,
    circular_orbit_speed,
    hohmann_transfer_speed,
    hohmann_phase_angle,
    gravitational_acceleration,
    hohmann_transfer_time
)
from integrators import velocity_verlet_step


import numpy as np

from bodies import Body
from physics import (
    G,
    circular_orbit_speed,
    hohmann_transfer_speed,
    hohmann_phase_angle,
    hohmann_transfer_time,
)
from integrators import velocity_verlet_step


# ---------------------------------------------------------
# 1. Physical and orbital parameters
# ---------------------------------------------------------

sun_mass = 1.0

earth_radius = 1.0
mars_radius = 1.53

earth_mass = 3.004e-6
mars_mass = 3.214e-7


# ---------------------------------------------------------
# 2. Circular orbital speeds
# ---------------------------------------------------------

earth_speed = circular_orbit_speed(
    central_mass=sun_mass,
    radius=earth_radius,
)

mars_speed = circular_orbit_speed(
    central_mass=sun_mass,
    radius=mars_radius,
)


# ---------------------------------------------------------
# 3. Hohmann transfer parameters
# ---------------------------------------------------------

ship_speed = hohmann_transfer_speed(
    central_mass=sun_mass,
    departure_radius=earth_radius,
    arrival_radius=mars_radius,
)

mars_phase = hohmann_phase_angle(
    central_mass=sun_mass,
    departure_radius=earth_radius,
    arrival_radius=mars_radius,
)

transfer_time = hohmann_transfer_time(
    central_mass=sun_mass,
    departure_radius=earth_radius,
    arrival_radius=mars_radius,
)


# ---------------------------------------------------------
# 4. Create the Sun
# ---------------------------------------------------------

sun = Body(
    name="Sun",
    mass=sun_mass,
    position=(0.0, 0.0),
    velocity=(0.0, 0.0),
)


# ---------------------------------------------------------
# 5. Acceleration used in the ideal Hohmann model
# ---------------------------------------------------------

def solar_acceleration(target, bodies):
    """
    Calculate the acceleration of a body due only to the Sun.

    This deliberately ignores the gravity of Earth and Mars so that
    the numerical trajectory can be compared with the analytical
    heliocentric Hohmann-transfer solution.

    Parameters
    ----------
    target : Body
        Body whose acceleration is calculated.

    bodies : list[Body]
        Included for compatibility with the integrator. It is not
        used in this idealized acceleration model.

    Returns
    -------
    numpy.ndarray
        Two-dimensional acceleration vector [ax, ay].
    """

    displacement = (
        sun.position - target.position
    )

    distance = np.linalg.norm(
        displacement
    )

    if distance == 0:
        raise ValueError(
            f"{target.name} occupies the same position as the Sun."
        )

    return (
        G
        * sun.mass
        * displacement
        / distance**3
    )


# ---------------------------------------------------------
# 6. Run one Hohmann-transfer simulation
# ---------------------------------------------------------

def run_transfer(number_of_steps):
    """
    Run an ideal heliocentric Earth-Mars Hohmann transfer.

    A fresh Earth, Mars and spacecraft are created for every run,
    ensuring that simulations with different time steps always start
    from identical initial conditions.

    Parameters
    ----------
    number_of_steps : int
        Number of Velocity Verlet steps used during the transfer.

    Returns
    -------
    tuple
        Time step in years,
        minimum spacecraft-Mars distance in AU,
        and time of closest approach in years.
    """

    # Create Earth
    earth = Body(
        name="Earth",
        mass=earth_mass,
        position=(earth_radius, 0.0),
        velocity=(0.0, earth_speed),
    )

    # Create Mars at the required Hohmann phase angle
    mars = Body(
        name="Mars",
        mass=mars_mass,
        position=(
            mars_radius * np.cos(mars_phase),
            mars_radius * np.sin(mars_phase),
        ),
        velocity=(
            -mars_speed * np.sin(mars_phase),
            mars_speed * np.cos(mars_phase),
        ),
    )

    # Treat the spacecraft as a massless test particle
    ship = Body(
        name="Ship",
        mass=0.0,
        position=(earth_radius, 0.0),
        velocity=(0.0, ship_speed),
    )

    bodies = [
        earth,
        mars,
        ship,
    ]

    # Choose dt so that the simulation ends exactly
    # at the analytical Hohmann transfer time.
    dt = (
        transfer_time
        / number_of_steps
    )

    minimum_distance = np.inf
    minimum_distance_time = 0.0

    # Propagate the system
    for step in range(number_of_steps):

        velocity_verlet_step(
            bodies,
            dt,
            solar_acceleration,
        )

        ship_mars_distance = np.linalg.norm(
            ship.position - mars.position
        )

        if ship_mars_distance < minimum_distance:

            minimum_distance = (
                ship_mars_distance
            )

            minimum_distance_time = (
                (step + 1) * dt
            )

    return (
        dt,
        minimum_distance,
        minimum_distance_time,
    )


# ---------------------------------------------------------
# 7. Compare numerical resolutions
# ---------------------------------------------------------

AU_TO_KM = 149_597_870.7

step_counts = [
    1500,
    3000,
    6000,
]


print(
    "Hohmann transfer time [years]:",
    transfer_time,
)

print(
    "Hohmann transfer time [days]:",
    transfer_time * 365.25,
)

print()

print(
    f"{'Steps':>10} "
    f"{'dt [years]':>15} "
    f"{'Min distance [km]':>20} "
    f"{'Closest approach [days]':>25}"
)


for number_of_steps in step_counts:

    (
        dt,
        minimum_distance,
        minimum_distance_time,
    ) = run_transfer(
        number_of_steps
    )

    minimum_distance_km = (
        minimum_distance
        * AU_TO_KM
    )

    closest_approach_days = (
        minimum_distance_time
        * 365.25
    )

    print(
        f"{number_of_steps:10d} "
        f"{dt:15.6e} "
        f"{minimum_distance_km:20.6f} "
        f"{closest_approach_days:25.6f}"
    )