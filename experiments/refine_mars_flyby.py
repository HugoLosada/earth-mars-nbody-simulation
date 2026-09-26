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

earth_radius_km = 6371.0
mars_radius_km = 3389.5


# ---------------------------------------------------------
# 3. Mission parameters
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

target_flyby_altitude_km = 300.0


# ---------------------------------------------------------
# 4. Hohmann reference
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
# 5. Earth escape
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
# 6. Numerical parameters
# ---------------------------------------------------------

escape_duration_days = 5.0

escape_dt_target = (
    1.25e-7
)

cruise_dt_target = (
    2.0e-6
)

mars_close_dt_target = (
    1.25e-7
)

mars_close_region_km = (
    1_000_000.0
)

simulation_end_time = (
    transfer_time
    + 30.0 / 365.25
)


# ---------------------------------------------------------
# 7. Create system
# ---------------------------------------------------------

def create_system(mars_phase_deg):

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
# 8. Run one transfer
# ---------------------------------------------------------

def run_transfer(mars_phase_deg):

    bodies = create_system(
        mars_phase_deg
    )

    mars = bodies[2]
    ship = bodies[3]

    current_time = 0.0

    minimum_distance_km = np.inf
    closest_approach_time = np.nan
    relative_speed_km_s = np.nan

    collision = False


    # -----------------------------------------------------
    # 8.1 Precise Earth escape
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

        current_time += escape_dt


    # -----------------------------------------------------
    # 8.2 Cruise and Mars encounter
    # -----------------------------------------------------

    previous_distance_km = np.inf

    while current_time < simulation_end_time:

        distance_before_km = (
            np.linalg.norm(
                ship.position
                - mars.position
            )
            * AU_TO_KM
        )


        # Choose time step.
        if (
            distance_before_km
            < mars_close_region_km
        ):

            dt = (
                mars_close_dt_target
            )

        else:

            dt = (
                cruise_dt_target
            )


        if (
            current_time + dt
            > simulation_end_time
        ):

            dt = (
                simulation_end_time
                - current_time
            )


        velocity_verlet_step(
            bodies,
            dt,
            gravitational_acceleration,
        )

        current_time += dt


        mars_distance_km = (
            np.linalg.norm(
                ship.position
                - mars.position
            )
            * AU_TO_KM
        )


        # -------------------------------------------------
        # Collision
        # -------------------------------------------------

        if (
            mars_distance_km
            <= mars_radius_km
        ):

            collision = True

            minimum_distance_km = (
                mars_distance_km
            )

            closest_approach_time = (
                current_time
            )

            relative_speed_km_s = (
                np.linalg.norm(
                    ship.velocity
                    - mars.velocity
                )
                * AU_TO_KM
                / YEAR_TO_SECONDS
            )

            break


        # -------------------------------------------------
        # New minimum
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

            relative_speed_km_s = (
                np.linalg.norm(
                    ship.velocity
                    - mars.velocity
                )
                * AU_TO_KM
                / YEAR_TO_SECONDS
            )


        # -------------------------------------------------
        # Closest approach already passed
        # -------------------------------------------------

        if (
            previous_distance_km
            < mars_close_region_km
            and mars_distance_km
            > previous_distance_km
        ):

            break


        previous_distance_km = (
            mars_distance_km
        )


    # -----------------------------------------------------
    # 8.3 Results
    # -----------------------------------------------------

    periapsis_altitude_km = (
        minimum_distance_km
        - mars_radius_km
    )

    closest_approach_days = (
        closest_approach_time
        * 365.25
    )


    return {
        "phase": mars_phase_deg,
        "collision": collision,
        "distance_km": minimum_distance_km,
        "altitude_km": periapsis_altitude_km,
        "time_days": closest_approach_days,
        "relative_speed_km_s": relative_speed_km_s,
    }


# ---------------------------------------------------------
# 9. Bisection setup
# ---------------------------------------------------------

lower_phase = 45.19
upper_phase = 45.20

# Stop once the phase interval is this small.
phase_tolerance_deg = 0.0001

# Also stop if the periapsis altitude is within
# this distance of the 300 km target.
altitude_tolerance_km = 10.0

max_iterations = 10


# ---------------------------------------------------------
# 10. Display setup
# ---------------------------------------------------------

print(
    "\n--- Mars flyby phase refinement ---"
)

print(
    "Target periapsis altitude [km]:",
    target_flyby_altitude_km,
)

print(
    "Initial phase interval [deg]:",
    lower_phase,
    "to",
    upper_phase,
)

print(
    "Phase tolerance [deg]:",
    phase_tolerance_deg,
)

print(
    "Altitude tolerance [km]:",
    altitude_tolerance_km,
)


# ---------------------------------------------------------
# 11. Bisection iterations
# ---------------------------------------------------------

best_safe_result = None


for iteration in range(
    1,
    max_iterations + 1,
):

    midpoint_phase = (
        0.5
        * (
            lower_phase
            + upper_phase
        )
    )

    print(
        "\n-----------------------------------------"
    )

    print(
        f"Iteration {iteration}"
    )

    print(
        "Testing Mars phase [deg]:",
        midpoint_phase,
    )

    result = run_transfer(
        midpoint_phase
    )


    if result["collision"]:

        print(
            "Status: COLLISION"
        )

        print(
            "Impact detected at centre distance [km]:",
            result["distance_km"],
        )

        # Larger phase moves this family of trajectories
        # deeper toward Mars, so move upper bound downward.
        upper_phase = (
            midpoint_phase
        )


    else:

        altitude_error = (
            result["altitude_km"]
            - target_flyby_altitude_km
        )

        print(
            "Status: SAFE FLYBY"
        )

        print(
            "Periapsis altitude [km]:",
            result["altitude_km"],
        )

        print(
            "Altitude error [km]:",
            altitude_error,
        )

        print(
            "Closest approach [days]:",
            result["time_days"],
        )

        print(
            "Relative speed [km/s]:",
            result["relative_speed_km_s"],
        )


        # Store the best physically safe trajectory.
        if (
            best_safe_result is None
            or abs(altitude_error)
            <
            abs(
                best_safe_result["altitude_km"]
                - target_flyby_altitude_km
            )
        ):

            best_safe_result = (
                result
            )


        # ---------------------------------------------
        # Bisection decision
        # ---------------------------------------------

        if abs(
            altitude_error
        ) <= altitude_tolerance_km:

            print(
                "\nTarget altitude reached "
                "within tolerance."
            )

            break


        # Altitude still too high:
        # increase phase toward Mars.
        if altitude_error > 0:

            lower_phase = (
                midpoint_phase
            )

        # Safe trajectory but altitude below 300 km:
        # decrease phase.
        else:

            upper_phase = (
                midpoint_phase
            )


    # -----------------------------------------------------
    # Check phase interval size
    # -----------------------------------------------------

    phase_interval = (
        upper_phase
        - lower_phase
    )

    print(
        "New phase interval [deg]:",
        lower_phase,
        "to",
        upper_phase,
    )

    print(
        "Interval width [deg]:",
        phase_interval,
    )


    if (
        phase_interval
        <= phase_tolerance_deg
    ):

        print(
            "\nPhase tolerance reached."
        )

        break


# ---------------------------------------------------------
# 12. Final best safe result
# ---------------------------------------------------------

print(
    "\n========================================="
)

print(
    "--- Best safe trajectory found ---"
)


if best_safe_result is not None:

    altitude_error = (
        best_safe_result["altitude_km"]
        - target_flyby_altitude_km
    )

    print(
        "Mars phase [deg]:",
        best_safe_result["phase"],
    )

    print(
        "Periapsis altitude [km]:",
        best_safe_result["altitude_km"],
    )

    print(
        "Target altitude [km]:",
        target_flyby_altitude_km,
    )

    print(
        "Altitude error [km]:",
        altitude_error,
    )

    print(
        "Closest approach [days]:",
        best_safe_result["time_days"],
    )

    print(
        "Relative speed at periapsis [km/s]:",
        best_safe_result["relative_speed_km_s"],
    )

    print(
        "Difference from analytical Hohmann phase [deg]:",
        best_safe_result["phase"]
        - np.degrees(
            analytical_mars_phase
        ),
    )

else:

    print(
        "No safe trajectory was found."
    )