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
# 4. Analytical Hohmann quantities
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
# 5. Trans-Mars Injection quantities
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
# 6. Hyperbolic escape geometry
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
# 7. Create a fresh system for a chosen Mars phase
# ---------------------------------------------------------

def create_system(mars_phase_deg):
    """
    Create a completely new Sun-Earth-Mars-spacecraft
    system for one candidate Mars phase angle.
    """

    mars_phase = np.radians(
        mars_phase_deg
    )


    # -----------------------------------------------------
    # Sun
    # -----------------------------------------------------

    sun = Body(
        name="Sun",
        mass=sun_mass,
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
    )


    # -----------------------------------------------------
    # Earth
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Mars
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Spacecraft position at hyperbolic periapsis
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Spacecraft velocity at hyperbolic periapsis
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Spacecraft
    # -----------------------------------------------------

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
# 8. Run one complete hybrid simulation
# ---------------------------------------------------------

def run_hybrid_transfer(
    mars_phase_deg,
):
    """
    Run one complete transfer using:

        3.94 s time steps during the first 5 days
        63.1 s time steps during interplanetary cruise

    Returns
    -------
    minimum_distance_km : float
        Minimum Ship-Mars distance.

    closest_approach_days : float
        Time of closest approach.

    relative_speed_km_s : float
        Ship-Mars relative speed at closest approach.
    """

    bodies = create_system(
        mars_phase_deg
    )

    mars = bodies[2]
    ship = bodies[3]


    # -----------------------------------------------------
    # 8.1 High-resolution Earth escape
    # -----------------------------------------------------

    escape_duration = (
        5.0 / 365.25
    )

    target_escape_dt = (
        1.25e-7
    )

    escape_steps = int(
        np.ceil(
            escape_duration
            / target_escape_dt
        )
    )

    escape_dt = (
        escape_duration
        / escape_steps
    )


    minimum_distance = np.inf
    closest_approach_time = 0.0
    relative_speed_at_closest = np.nan


    for step in range(escape_steps):

        velocity_verlet_step(
            bodies,
            escape_dt,
            gravitational_acceleration,
        )

        current_time = (
            (step + 1)
            * escape_dt
        )

        mars_distance = np.linalg.norm(
            ship.position
            - mars.position
        )

        if mars_distance < minimum_distance:

            minimum_distance = (
                mars_distance
            )

            closest_approach_time = (
                current_time
            )

            relative_speed_at_closest = (
                np.linalg.norm(
                    ship.velocity
                    - mars.velocity
                )
            )


    # -----------------------------------------------------
    # 8.2 Interplanetary cruise
    # -----------------------------------------------------

    total_simulation_time = (
        transfer_time
        + 30.0 / 365.25
    )

    cruise_duration = (
        total_simulation_time
        - escape_duration
    )

    target_cruise_dt = (
        2.0e-6
    )

    cruise_steps = int(
        np.ceil(
            cruise_duration
            / target_cruise_dt
        )
    )

    cruise_dt = (
        cruise_duration
        / cruise_steps
    )


    for step in range(cruise_steps):

        velocity_verlet_step(
            bodies,
            cruise_dt,
            gravitational_acceleration,
        )

        current_time = (
            escape_duration
            + (step + 1)
            * cruise_dt
        )

        mars_distance = np.linalg.norm(
            ship.position
            - mars.position
        )

        if mars_distance < minimum_distance:

            minimum_distance = (
                mars_distance
            )

            closest_approach_time = (
                current_time
            )

            relative_speed_at_closest = (
                np.linalg.norm(
                    ship.velocity
                    - mars.velocity
                )
            )


    # -----------------------------------------------------
    # 8.3 Convert results
    # -----------------------------------------------------

    minimum_distance_km = (
        minimum_distance
        * AU_TO_KM
    )

    closest_approach_days = (
        closest_approach_time
        * 365.25
    )

    relative_speed_km_s = (
        relative_speed_at_closest
        * AU_TO_KM
        / YEAR_TO_SECONDS
    )


    return (
        minimum_distance_km,
        closest_approach_days,
        relative_speed_km_s,
    )


# ---------------------------------------------------------
# 9. Candidate Mars phases
# ---------------------------------------------------------

candidate_phases = [
    45.00,
    45.05,
    45.10,
    45.15,
    45.20,
]


# ---------------------------------------------------------
# 10. Display reference values
# ---------------------------------------------------------

print(
    "\n--- Hybrid Mars phase optimization ---",
    flush=True,
)

print(
    "Analytical Hohmann phase [deg]:",
    np.degrees(
        analytical_mars_phase
    ),
    flush=True,
)

print(
    "Analytical Hohmann transfer time [days]:",
    transfer_time * 365.25,
    flush=True,
)

print(
    "\nEscape time step [s]:",
    1.25e-7 * YEAR_TO_SECONDS,
    flush=True,
)

print(
    "Cruise time step [s]:",
    2.0e-6 * YEAR_TO_SECONDS,
    flush=True,
)

print(
    "\nCandidate phases:",
    candidate_phases,
    flush=True,
)


# ---------------------------------------------------------
# 11. Run phase search
# ---------------------------------------------------------

results = []


for simulation_number, phase in enumerate(
    candidate_phases,
    start=1,
):

    print(
        "\n-----------------------------------------",
        flush=True,
    )

    print(
        f"Simulation "
        f"{simulation_number}/"
        f"{len(candidate_phases)}",
        flush=True,
    )

    print(
        "Mars phase [deg]:",
        phase,
        flush=True,
    )

    print(
        "Running high-resolution escape...",
        flush=True,
    )


    (
        minimum_distance_km,
        closest_approach_days,
        relative_speed_km_s,
    ) = run_hybrid_transfer(
        mars_phase_deg=phase,
    )


    results.append(
        (
            phase,
            minimum_distance_km,
            closest_approach_days,
            relative_speed_km_s,
        )
    )


    print(
        "Simulation completed.",
        flush=True,
    )

    print(
        "Minimum Ship-Mars distance [km]:",
        minimum_distance_km,
        flush=True,
    )

    print(
        "Closest approach [days]:",
        closest_approach_days,
        flush=True,
    )

    print(
        "Relative speed at closest approach [km/s]:",
        relative_speed_km_s,
        flush=True,
    )


# ---------------------------------------------------------
# 12. Summary table
# ---------------------------------------------------------

print(
    "\n--- Phase-search results ---"
)

print(
    f"{'Phase [deg]':>12} "
    f"{'Min distance [km]':>20} "
    f"{'Closest approach [days]':>25} "
    f"{'Relative speed [km/s]':>25}"
)


for result in results:

    phase = result[0]
    minimum_distance_km = result[1]
    closest_approach_days = result[2]
    relative_speed_km_s = result[3]

    print(
        f"{phase:12.3f} "
        f"{minimum_distance_km:20.3f} "
        f"{closest_approach_days:25.6f} "
        f"{relative_speed_km_s:25.6f}"
    )


# ---------------------------------------------------------
# 13. Find best candidate
# ---------------------------------------------------------

best_result = min(
    results,
    key=lambda result: result[1],
)


best_phase = (
    best_result[0]
)

best_distance = (
    best_result[1]
)

best_time = (
    best_result[2]
)

best_relative_speed = (
    best_result[3]
)


# ---------------------------------------------------------
# 14. Display best result
# ---------------------------------------------------------

print(
    "\n--- Best hybrid phase candidate ---"
)

print(
    "Best Mars phase [deg]:",
    best_phase,
)

print(
    "Minimum Ship-Mars distance [km]:",
    best_distance,
)

print(
    "Closest approach [days]:",
    best_time,
)

print(
    "Relative speed at closest approach [km/s]:",
    best_relative_speed,
)

print(
    "Difference from analytical Hohmann phase [deg]:",
    best_phase
    - np.degrees(
        analytical_mars_phase
    ),
)