import numpy as np

from bodies import Body

from physics import (
    G,
    circular_orbit_speed,
    hohmann_transfer_speed,
    hohmann_phase_angle,
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
# 3. Planetary radii
# ---------------------------------------------------------

earth_radius_km = 6371.0

# Approximate mean radius of Mars.
mars_radius_km = 3389.5


# ---------------------------------------------------------
# 4. Desired Mars flyby
# ---------------------------------------------------------

target_flyby_altitude_km = 300.0

target_mars_distance_km = (
    mars_radius_km
    + target_flyby_altitude_km
)


# ---------------------------------------------------------
# 5. Earth parking orbit
# ---------------------------------------------------------

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
# 6. Analytical Hohmann quantities
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

analytical_mars_phase = hohmann_phase_angle(
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
# 7. Trans-Mars Injection quantities
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


# ---------------------------------------------------------
# 8. Hyperbolic Earth-escape geometry
# ---------------------------------------------------------

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
# 9. Numerical parameters
# ---------------------------------------------------------

# Precise integration close to Earth.
escape_duration_days = 5.0

escape_dt_target = (
    1.25e-7
)

# Validated cruise time step.
cruise_dt_target = (
    2.0e-6
)

# When the spacecraft comes within this distance of Mars,
# reduce the time step again.
mars_close_region_km = (
    1_000_000.0
)

# Fine time step near Mars.
mars_close_dt_target = (
    1.25e-7
)

# End the simulation 30 days after the analytical
# Hohmann arrival time.
simulation_end_time = (
    transfer_time
    + 30.0 / 365.25
)


# ---------------------------------------------------------
# 10. Create system
# ---------------------------------------------------------

def create_system(mars_phase_deg):
    """
    Create a new Sun-Earth-Mars-spacecraft system
    for a chosen initial Mars phase angle.
    """

    mars_phase = np.radians(
        mars_phase_deg
    )

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
# 11. Run one transfer
# ---------------------------------------------------------

def run_transfer(mars_phase_deg):
    """
    Run one Earth-Mars transfer.

    Three numerical regimes are used:

    1. Fine time step during Earth escape.
    2. Coarser validated step during heliocentric cruise.
    3. Fine time step again close to Mars.

    Returns
    -------
    dict
        Simulation results.
    """

    bodies = create_system(
        mars_phase_deg
    )

    mars = bodies[2]
    ship = bodies[3]

    current_time = 0.0

    minimum_distance_km = np.inf
    closest_approach_time = 0.0
    relative_speed_at_closest_km_s = np.nan

    collision = False
    collision_time_days = np.nan


    # -----------------------------------------------------
    # 11.1 Earth escape
    # -----------------------------------------------------

    escape_duration = (
        escape_duration_days
        / 365.25
    )

    escape_steps = int(
        np.ceil(
            escape_duration
            / escape_dt_target
        )
    )

    escape_dt = (
        escape_duration
        / escape_steps
    )

    for step in range(escape_steps):

        velocity_verlet_step(
            bodies,
            escape_dt,
            gravitational_acceleration,
        )

        current_time += (
            escape_dt
        )


    # -----------------------------------------------------
    # 11.2 Cruise + Mars encounter
    # -----------------------------------------------------

    previous_mars_distance_km = np.inf

    passed_closest_approach = False


    while current_time < simulation_end_time:

        mars_distance_km_before = (
            np.linalg.norm(
                ship.position
                - mars.position
            )
            * AU_TO_KM
        )


        # -------------------------------------------------
        # Select time step
        # -------------------------------------------------

        if (
            mars_distance_km_before
            < mars_close_region_km
        ):

            dt = (
                mars_close_dt_target
            )

        else:

            dt = (
                cruise_dt_target
            )


        # Do not step beyond the requested final time.
        if (
            current_time + dt
            > simulation_end_time
        ):

            dt = (
                simulation_end_time
                - current_time
            )


        # -------------------------------------------------
        # Advance the system
        # -------------------------------------------------

        velocity_verlet_step(
            bodies,
            dt,
            gravitational_acceleration,
        )

        current_time += dt


        # -------------------------------------------------
        # Mars distance after the step
        # -------------------------------------------------

        mars_distance_km = (
            np.linalg.norm(
                ship.position
                - mars.position
            )
            * AU_TO_KM
        )


        # -------------------------------------------------
        # Collision detection
        # -------------------------------------------------

        if (
            mars_distance_km
            <= mars_radius_km
        ):

            collision = True

            collision_time_days = (
                current_time
                * 365.25
            )

            minimum_distance_km = (
                mars_distance_km
            )

            closest_approach_time = (
                current_time
            )

            relative_speed_at_closest_km_s = (
                np.linalg.norm(
                    ship.velocity
                    - mars.velocity
                )
                * AU_TO_KM
                / YEAR_TO_SECONDS
            )

            break


        # -------------------------------------------------
        # Record closest safe approach
        # -------------------------------------------------

        if (
            mars_distance_km
            < minimum_distance_km
        ):

            minimum_distance_km = (
                mars_distance_km
            )

            closest_approach_time = (
                current_time
            )

            relative_speed_at_closest_km_s = (
                np.linalg.norm(
                    ship.velocity
                    - mars.velocity
                )
                * AU_TO_KM
                / YEAR_TO_SECONDS
            )


        # -------------------------------------------------
        # Detect whether the spacecraft has passed Mars
        # -------------------------------------------------

        if (
            mars_distance_km
            > previous_mars_distance_km
            and minimum_distance_km
            < mars_close_region_km
        ):

            passed_closest_approach = True


        previous_mars_distance_km = (
            mars_distance_km
        )


        # -------------------------------------------------
        # Once the encounter is over and the spacecraft
        # has moved far from Mars again, stop.
        # -------------------------------------------------

        if (
            passed_closest_approach
            and mars_distance_km
            > mars_close_region_km
        ):

            break


    # -----------------------------------------------------
    # 11.3 Derived quantities
    # -----------------------------------------------------

    closest_approach_days = (
        closest_approach_time
        * 365.25
    )

    periapsis_altitude_km = (
        minimum_distance_km
        - mars_radius_km
    )


    # -----------------------------------------------------
    # 11.4 Classify the trajectory
    # -----------------------------------------------------

    if collision:

        status = "COLLISION"

        objective_error_km = np.inf

    else:

        status = "SAFE FLYBY"

        objective_error_km = abs(
            minimum_distance_km
            - target_mars_distance_km
        )


    return {
        "phase": mars_phase_deg,
        "status": status,
        "minimum_distance_km": minimum_distance_km,
        "periapsis_altitude_km": periapsis_altitude_km,
        "closest_approach_days": closest_approach_days,
        "relative_speed_km_s": relative_speed_at_closest_km_s,
        "objective_error_km": objective_error_km,
        "collision_time_days": collision_time_days,
    }


# ---------------------------------------------------------
# 12. Candidate phases
# ---------------------------------------------------------

# 45.15 degrees was already a safe but distant flyby.
# 45.20 degrees previously entered Mars.
#
# We now explore the transition between them.

candidate_phases = [
    45.16,
    45.17,
    45.18,
    45.19,
    45.20,
]


# ---------------------------------------------------------
# 13. Display setup
# ---------------------------------------------------------

print(
    "\n--- Mars flyby optimization ---"
)

print(
    "Analytical Hohmann phase [deg]:",
    np.degrees(
        analytical_mars_phase
    ),
)

print(
    "Mars radius [km]:",
    mars_radius_km,
)

print(
    "Target flyby altitude [km]:",
    target_flyby_altitude_km,
)

print(
    "Target distance from Mars centre [km]:",
    target_mars_distance_km,
)

print(
    "Earth escape dt [s]:",
    escape_dt_target
    * YEAR_TO_SECONDS,
)

print(
    "Cruise dt [s]:",
    cruise_dt_target
    * YEAR_TO_SECONDS,
)

print(
    "Close-Mars dt [s]:",
    mars_close_dt_target
    * YEAR_TO_SECONDS,
)

print(
    "Candidate phases:",
    candidate_phases,
)


# ---------------------------------------------------------
# 14. Run simulations
# ---------------------------------------------------------

results = []


for simulation_number, phase in enumerate(
    candidate_phases,
    start=1,
):

    print(
        "\n-----------------------------------------"
    )

    print(
        f"Simulation "
        f"{simulation_number}/"
        f"{len(candidate_phases)}"
    )

    print(
        "Mars phase [deg]:",
        phase,
    )

    result = run_transfer(
        mars_phase_deg=phase
    )

    results.append(
        result
    )

    print(
        "Status:",
        result["status"],
    )

    print(
        "Minimum centre distance [km]:",
        result["minimum_distance_km"],
    )

    print(
        "Periapsis altitude [km]:",
        result["periapsis_altitude_km"],
    )

    print(
        "Closest approach [days]:",
        result["closest_approach_days"],
    )

    print(
        "Relative speed [km/s]:",
        result["relative_speed_km_s"],
    )

    if (
        result["status"]
        == "SAFE FLYBY"
    ):

        print(
            "Error from 300 km target [km]:",
            result["objective_error_km"],
        )

    else:

        print(
            "Impact detected."
        )


# ---------------------------------------------------------
# 15. Results table
# ---------------------------------------------------------

print(
    "\n--- Flyby search results ---"
)

print(
    f"{'Phase':>8} "
    f"{'Status':>14} "
    f"{'Altitude [km]':>16} "
    f"{'Time [days]':>15} "
    f"{'Rel speed [km/s]':>20} "
    f"{'Target error [km]':>20}"
)


for result in results:

    if np.isfinite(
        result["objective_error_km"]
    ):

        error_text = (
            f'{result["objective_error_km"]:.3f}'
        )

    else:

        error_text = (
            "N/A"
        )


    print(
        f'{result["phase"]:8.3f} '
        f'{result["status"]:>14} '
        f'{result["periapsis_altitude_km"]:16.3f} '
        f'{result["closest_approach_days"]:15.6f} '
        f'{result["relative_speed_km_s"]:20.6f} '
        f'{error_text:>20}'
    )


# ---------------------------------------------------------
# 16. Find best SAFE trajectory
# ---------------------------------------------------------

safe_results = [
    result
    for result in results
    if result["status"]
    == "SAFE FLYBY"
]


if len(safe_results) > 0:

    best_result = min(
        safe_results,
        key=lambda result:
            result["objective_error_km"],
    )

    print(
        "\n--- Best safe flyby candidate ---"
    )

    print(
        "Mars phase [deg]:",
        best_result["phase"],
    )

    print(
        "Periapsis altitude [km]:",
        best_result["periapsis_altitude_km"],
    )

    print(
        "Target altitude [km]:",
        target_flyby_altitude_km,
    )

    print(
        "Altitude error [km]:",
        best_result["objective_error_km"],
    )

    print(
        "Closest approach [days]:",
        best_result["closest_approach_days"],
    )

    print(
        "Relative speed [km/s]:",
        best_result["relative_speed_km_s"],
    )

    print(
        "Difference from analytical Hohmann phase [deg]:",
        best_result["phase"]
        - np.degrees(
            analytical_mars_phase
        ),
    )

else:

    print(
        "\nNo safe flyby was found "
        "in the tested phase range."
    )