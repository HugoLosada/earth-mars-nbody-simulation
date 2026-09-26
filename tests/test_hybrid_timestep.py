import numpy as np

from bodies import Body

from physics import (
    G,
    circular_orbit_speed,
    hohmann_transfer_speed,
    hohmann_transfer_time,
    hyperbolic_periapsis_speed,
    hyperbolic_eccentricity,
    hyperbolic_asymptote_angle,
    gravitational_acceleration,
)

from integrators import velocity_verlet_step


# ---------------------------------------------------------
# 1. Unit conversions
# ---------------------------------------------------------

AU_TO_KM = 149_597_870.7
YEAR_TO_SECONDS = 365.25 * 24 * 3600


# ---------------------------------------------------------
# 2. Physical parameters
# ---------------------------------------------------------

sun_mass = 1.0

earth_mass = 3.004e-6
mars_mass = 3.214e-7

earth_orbit_radius = 1.0
mars_orbit_radius = 1.53


# ---------------------------------------------------------
# 3. Earth parking orbit
# ---------------------------------------------------------

earth_radius_km = 6371.0
parking_altitude_km = 300.0

parking_radius_km = (
    earth_radius_km
    + parking_altitude_km
)

parking_radius = (
    parking_radius_km
    / AU_TO_KM
)


# ---------------------------------------------------------
# 4. Analytical heliocentric quantities
# ---------------------------------------------------------

earth_speed = circular_orbit_speed(
    central_mass=sun_mass,
    radius=earth_orbit_radius,
)

mars_speed = circular_orbit_speed(
    central_mass=sun_mass,
    radius=mars_orbit_radius,
)

hohmann_departure_speed = hohmann_transfer_speed(
    central_mass=sun_mass,
    departure_radius=earth_orbit_radius,
    arrival_radius=mars_orbit_radius,
)

transfer_time = hohmann_transfer_time(
    central_mass=sun_mass,
    departure_radius=earth_orbit_radius,
    arrival_radius=mars_orbit_radius,
)


# ---------------------------------------------------------
# 5. Hyperbolic escape quantities
# ---------------------------------------------------------

v_infinity = (
    hohmann_departure_speed
    - earth_speed
)

mu_earth = (
    G * earth_mass
)

injection_speed = hyperbolic_periapsis_speed(
    mu=mu_earth,
    periapsis_radius=parking_radius,
    v_infinity=v_infinity,
)

hyperbolic_e = hyperbolic_eccentricity(
    mu=mu_earth,
    periapsis_radius=parking_radius,
    v_infinity=v_infinity,
)

asymptote_true_anomaly = hyperbolic_asymptote_angle(
    hyperbolic_e
)

desired_v_infinity_angle = (
    np.pi / 2
)

periapsis_angle = (
    desired_v_infinity_angle
    - asymptote_true_anomaly
)


# ---------------------------------------------------------
# 6. Mars phase used for this numerical test
# ---------------------------------------------------------

# This is still only our current candidate.
# It is NOT yet the final optimized phase.
mars_phase_deg = 45.1

mars_phase = np.radians(
    mars_phase_deg
)


# ---------------------------------------------------------
# 7. Create initial system
# ---------------------------------------------------------

def create_initial_system():
    """
    Create a fresh Sun-Earth-Mars-spacecraft system.
    """

    sun = Body(
        name="Sun",
        mass=sun_mass,
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
    )

    earth = Body(
        name="Earth",
        mass=earth_mass,
        position=(
            earth_orbit_radius,
            0.0,
        ),
        velocity=(
            0.0,
            earth_speed,
        ),
    )

    mars = Body(
        name="Mars",
        mass=mars_mass,
        position=(
            mars_orbit_radius
            * np.cos(mars_phase),

            mars_orbit_radius
            * np.sin(mars_phase),
        ),
        velocity=(
            -mars_speed
            * np.sin(mars_phase),

            mars_speed
            * np.cos(mars_phase),
        ),
    )

    ship_relative_position = np.array(
        [
            parking_radius
            * np.cos(periapsis_angle),

            parking_radius
            * np.sin(periapsis_angle),
        ]
    )

    ship_position = (
        earth.position
        + ship_relative_position
    )

    ship_relative_velocity = (
        injection_speed
        * np.array(
            [
                -np.sin(periapsis_angle),
                np.cos(periapsis_angle),
            ]
        )
    )

    ship_velocity = (
        earth.velocity
        + ship_relative_velocity
    )

    ship = Body(
        name="Ship",
        mass=0.0,
        position=ship_position,
        velocity=ship_velocity,
    )

    return [
        sun,
        earth,
        mars,
        ship,
    ]


# ---------------------------------------------------------
# 8. Clone a system
# ---------------------------------------------------------

def clone_system(bodies):
    """
    Create an independent copy of every body.

    This allows each cruise simulation to start
    from exactly the same state.
    """

    cloned_bodies = []

    for body in bodies:

        cloned_body = Body(
            name=body.name,
            mass=body.mass,
            position=body.position.copy(),
            velocity=body.velocity.copy(),
        )

        cloned_bodies.append(
            cloned_body
        )

    return cloned_bodies


# ---------------------------------------------------------
# 9. Precise Earth-escape phase
# ---------------------------------------------------------

def propagate_escape(bodies):
    """
    Propagate the first five days using a small time step.

    The small step is necessary because the spacecraft
    experiences strong and rapidly changing Earth gravity
    close to periapsis.
    """

    escape_duration = (
        5.0 / 365.25
    )

    target_escape_dt = (
        1.25e-7
    )

    number_of_steps = int(
        np.ceil(
            escape_duration
            / target_escape_dt
        )
    )

    # Adjust dt so the propagation finishes
    # at exactly five days.
    dt = (
        escape_duration
        / number_of_steps
    )

    minimum_distance = np.inf
    minimum_distance_time = 0.0

    mars = bodies[2]
    ship = bodies[3]

    print(
        "\n--- Earth escape phase ---",
        flush=True,
    )

    print(
        "Escape dt [s]:",
        dt * YEAR_TO_SECONDS,
        flush=True,
    )

    print(
        "Escape steps:",
        number_of_steps,
        flush=True,
    )

    for step in range(number_of_steps):

        velocity_verlet_step(
            bodies,
            dt,
            gravitational_acceleration,
        )

        current_time = (
            (step + 1)
            * dt
        )

        mars_distance = np.linalg.norm(
            ship.position
            - mars.position
        )

        if mars_distance < minimum_distance:

            minimum_distance = (
                mars_distance
            )

            minimum_distance_time = (
                current_time
            )

        # Show progress every 25,000 steps.
        if (
            (step + 1) % 25_000 == 0
            or step + 1 == number_of_steps
        ):

            progress_percent = (
                100
                * (step + 1)
                / number_of_steps
            )

            print(
                f"Escape progress: "
                f"{step + 1}/{number_of_steps} "
                f"({progress_percent:.1f}%)",
                flush=True,
            )

    return (
        bodies,
        minimum_distance,
        minimum_distance_time,
    )


# ---------------------------------------------------------
# 10. Interplanetary cruise
# ---------------------------------------------------------

def propagate_cruise(
    initial_bodies,
    target_cruise_dt,
    initial_minimum_distance,
    initial_minimum_time,
):
    """
    Propagate the mission from day 5 until the end.

    Every cruise simulation starts from the same
    high-resolution escape state.
    """

    bodies = clone_system(
        initial_bodies
    )

    mars = bodies[2]
    ship = bodies[3]

    escape_duration = (
        5.0 / 365.25
    )

    total_simulation_time = (
        transfer_time
        + 30.0 / 365.25
    )

    cruise_duration = (
        total_simulation_time
        - escape_duration
    )

    number_of_steps = int(
        np.ceil(
            cruise_duration
            / target_cruise_dt
        )
    )

    # Adjust dt so all simulations finish
    # at exactly the same final time.
    dt = (
        cruise_duration
        / number_of_steps
    )

    minimum_distance = (
        initial_minimum_distance
    )

    minimum_distance_time = (
        initial_minimum_time
    )

    print(
        "\nCruise integration:",
        flush=True,
    )

    print(
        "Actual cruise dt [s]:",
        dt * YEAR_TO_SECONDS,
        flush=True,
    )

    print(
        "Cruise steps:",
        number_of_steps,
        flush=True,
    )

    for step in range(number_of_steps):

        velocity_verlet_step(
            bodies,
            dt,
            gravitational_acceleration,
        )

        current_time = (
            escape_duration
            + (step + 1) * dt
        )

        mars_distance = np.linalg.norm(
            ship.position
            - mars.position
        )

        if mars_distance < minimum_distance:

            minimum_distance = (
                mars_distance
            )

            minimum_distance_time = (
                current_time
            )

        # Show progress every 100,000 iterations.
        if (
            (step + 1) % 100_000 == 0
            or step + 1 == number_of_steps
        ):

            progress_percent = (
                100
                * (step + 1)
                / number_of_steps
            )

            print(
                f"Cruise progress: "
                f"{step + 1}/{number_of_steps} "
                f"({progress_percent:.1f}%)",
                flush=True,
            )

    minimum_distance_km = (
        minimum_distance
        * AU_TO_KM
    )

    closest_approach_days = (
        minimum_distance_time
        * 365.25
    )

    final_ship_position = (
        ship.position.copy()
    )

    final_ship_velocity = (
        ship.velocity.copy()
    )

    return (
        dt,
        number_of_steps,
        minimum_distance_km,
        closest_approach_days,
        final_ship_position,
        final_ship_velocity,
    )


# ---------------------------------------------------------
# 11. Create one high-resolution escape trajectory
# ---------------------------------------------------------

print(
    "\nCreating initial system...",
    flush=True,
)

initial_bodies = (
    create_initial_system()
)

(
    escaped_bodies,
    escape_minimum_distance,
    escape_minimum_time,
) = propagate_escape(
    initial_bodies
)

print(
    "\nEarth escape completed.",
    flush=True,
)


# ---------------------------------------------------------
# 12. Cruise time steps to test
# ---------------------------------------------------------

# We only compare 63 s and 31.6 s for now.
#
# The previous 15.8 s simulation was responsible
# for most of the computation time.

cruise_time_steps = [
    2.0e-6,
    1.0e-6,
]


# ---------------------------------------------------------
# 13. Run hybrid time-step comparison
# ---------------------------------------------------------

print(
    "\n--- Hybrid cruise time-step convergence ---",
    flush=True,
)

print(
    "Mars phase [deg]:",
    mars_phase_deg,
    flush=True,
)

results = []


for simulation_number, target_dt in enumerate(
    cruise_time_steps,
    start=1,
):

    print(
        "\n-----------------------------------------",
        flush=True,
    )

    print(
        f"Starting cruise simulation "
        f"{simulation_number}/"
        f"{len(cruise_time_steps)}",
        flush=True,
    )

    print(
        "Requested cruise dt [s]:",
        target_dt
        * YEAR_TO_SECONDS,
        flush=True,
    )

    result = propagate_cruise(
        initial_bodies=escaped_bodies,
        target_cruise_dt=target_dt,
        initial_minimum_distance=escape_minimum_distance,
        initial_minimum_time=escape_minimum_time,
    )

    results.append(
        result
    )

    (
        actual_dt,
        number_of_steps,
        minimum_distance_km,
        closest_approach_days,
        final_position,
        final_velocity,
    ) = result

    print(
        "\nSimulation completed.",
        flush=True,
    )

    print(
        "Minimum Ship-Mars distance [km]:",
        minimum_distance_km,
        flush=True,
    )

    print(
        "Closest approach time [days]:",
        closest_approach_days,
        flush=True,
    )


# ---------------------------------------------------------
# 14. Summary table
# ---------------------------------------------------------

print(
    "\n--- Hybrid cruise results ---"
)

print(
    f"{'Cruise dt [s]':>15} "
    f"{'Cruise steps':>15} "
    f"{'Min dist [km]':>18} "
    f"{'Closest approach [days]':>25}"
)


for result in results:

    (
        actual_dt,
        number_of_steps,
        minimum_distance_km,
        closest_approach_days,
        final_position,
        final_velocity,
    ) = result

    print(
        f"{actual_dt * YEAR_TO_SECONDS:15.3f} "
        f"{number_of_steps:15d} "
        f"{minimum_distance_km:18.3f} "
        f"{closest_approach_days:25.6f}"
    )


# ---------------------------------------------------------
# 15. Compare the two cruise resolutions
# ---------------------------------------------------------

coarse = results[0]
fine = results[1]


coarse_dt_seconds = (
    coarse[0]
    * YEAR_TO_SECONDS
)

fine_dt_seconds = (
    fine[0]
    * YEAR_TO_SECONDS
)


# Difference in closest-approach distance
minimum_distance_difference = abs(
    coarse[2]
    - fine[2]
)


# Difference in closest-approach time
closest_time_difference = abs(
    coarse[3]
    - fine[3]
)


# Difference in final spacecraft position
final_position_difference_km = (
    np.linalg.norm(
        coarse[4]
        - fine[4]
    )
    * AU_TO_KM
)


# Difference in final spacecraft velocity
final_velocity_difference_km_s = (
    np.linalg.norm(
        coarse[5]
        - fine[5]
    )
    * AU_TO_KM
    / YEAR_TO_SECONDS
)


# ---------------------------------------------------------
# 16. Display comparison
# ---------------------------------------------------------

print(
    "\n--- Cruise time-step difference ---"
)

print(
    "Cruise dt:",
    coarse_dt_seconds,
    "s ->",
    fine_dt_seconds,
    "s",
)

print(
    "Closest-approach distance difference [km]:",
    minimum_distance_difference,
)

print(
    "Closest-approach time difference [days]:",
    closest_time_difference,
)

print(
    "Final position difference [km]:",
    final_position_difference_km,
)

print(
    "Final velocity difference [km/s]:",
    final_velocity_difference_km_s,
)