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


target_mars_altitude_km = 300.0

altitude_tolerance_km = 10.0


# ---------------------------------------------------------
# 4. Final Mars phase candidate
# ---------------------------------------------------------

# Obtained by interpolating between the two closest
# previously simulated trajectories:
#
# 45.196953125 deg -> 316.406 km
# 45.197031250 deg -> 250.351 km
#
# This is NOT accepted blindly.
# We will verify it with a new simulation.

mars_phase_deg = 45.1969725


# ---------------------------------------------------------
# 5. Analytical Hohmann reference
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
# 6. Earth escape parameters
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
# 7. Numerical parameters
# ---------------------------------------------------------

escape_duration_days = 5.0

# Validated Earth-escape step:
# approximately 3.94 seconds
escape_dt_target = (
    1.25e-7
)

# Validated heliocentric cruise step:
# approximately 63.1 seconds
cruise_dt_target = (
    2.0e-6
)

# Validated close-Mars step:
# approximately 3.94 seconds
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
# 8. Create the initial system
# ---------------------------------------------------------

def create_system():
    """
    Create the final candidate mission configuration.
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


    # -----------------------------------------------------
    # Spacecraft position at Earth hyperbolic periapsis
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
    # Spacecraft Earth-relative injection velocity
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
# 9. Run final mission
# ---------------------------------------------------------

def run_final_mission():

    bodies = create_system()

    earth = bodies[1]
    mars = bodies[2]
    ship = bodies[3]

    current_time = 0.0


    minimum_mars_distance_km = np.inf
    closest_approach_time = np.nan
    relative_speed_at_periapsis_km_s = np.nan

    collision = False


    # -----------------------------------------------------
    # 9.1 Precise Earth escape
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


    print(
        "\n--- Phase 1: Earth escape ---",
        flush=True,
    )

    print(
        "dt [s]:",
        escape_dt * YEAR_TO_SECONDS,
        flush=True,
    )

    print(
        "Steps:",
        escape_steps,
        flush=True,
    )


    for step in range(escape_steps):

        velocity_verlet_step(
            bodies,
            escape_dt,
            gravitational_acceleration,
        )

        current_time += escape_dt


        if (
            (step + 1) % 25_000 == 0
            or step + 1 == escape_steps
        ):

            progress = (
                100
                * (step + 1)
                / escape_steps
            )

            print(
                f"Progress: {progress:.1f}%",
                flush=True,
            )


    # State after Earth escape
    earth_ship_distance_km = (
        np.linalg.norm(
            ship.position
            - earth.position
        )
        * AU_TO_KM
    )

    earth_relative_speed_km_s = (
        np.linalg.norm(
            ship.velocity
            - earth.velocity
        )
        * AU_TO_KM
        / YEAR_TO_SECONDS
    )


    # -----------------------------------------------------
    # 9.2 Heliocentric cruise
    # -----------------------------------------------------

    print(
        "\n--- Phase 2: Interplanetary cruise ---",
        flush=True,
    )

    cruise_steps = 0


    while current_time < simulation_end_time:

        mars_distance_km = (
            np.linalg.norm(
                ship.position
                - mars.position
            )
            * AU_TO_KM
        )


        if (
            mars_distance_km
            <= mars_close_region_km
        ):

            break


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

        cruise_steps += 1


        if (
            cruise_steps % 100_000 == 0
        ):

            mars_distance_km = (
                np.linalg.norm(
                    ship.position
                    - mars.position
                )
                * AU_TO_KM
            )

            print(
                f"Cruise steps: "
                f"{cruise_steps} | "
                f"Mars distance: "
                f"{mars_distance_km:,.0f} km",
                flush=True,
            )


    print(
        "Mars encounter region reached.",
        flush=True,
    )

    print(
        "Mission time [days]:",
        current_time * 365.25,
        flush=True,
    )


    # -----------------------------------------------------
    # 9.3 Close Mars encounter
    # -----------------------------------------------------

    print(
        "\n--- Phase 3: Mars encounter ---",
        flush=True,
    )

    print(
        "Close-Mars dt [s]:",
        mars_close_dt_target
        * YEAR_TO_SECONDS,
        flush=True,
    )


    previous_mars_distance_km = (
        np.linalg.norm(
            ship.position
            - mars.position
        )
        * AU_TO_KM
    )

    encounter_steps = 0


    while current_time < simulation_end_time:

        dt = (
            mars_close_dt_target
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

        encounter_steps += 1


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

            minimum_mars_distance_km = (
                mars_distance_km
            )

            closest_approach_time = (
                current_time
            )

            relative_speed_at_periapsis_km_s = (
                np.linalg.norm(
                    ship.velocity
                    - mars.velocity
                )
                * AU_TO_KM
                / YEAR_TO_SECONDS
            )

            break


        # -------------------------------------------------
        # New minimum distance
        # -------------------------------------------------

        if (
            mars_distance_km
            < minimum_mars_distance_km
        ):

            minimum_mars_distance_km = (
                mars_distance_km
            )

            closest_approach_time = (
                current_time
            )

            relative_speed_at_periapsis_km_s = (
                np.linalg.norm(
                    ship.velocity
                    - mars.velocity
                )
                * AU_TO_KM
                / YEAR_TO_SECONDS
            )


        # -------------------------------------------------
        # Periapsis has been passed
        # -------------------------------------------------

        if (
            mars_distance_km
            > previous_mars_distance_km
            and np.isfinite(
                minimum_mars_distance_km
            )
        ):

            break


        previous_mars_distance_km = (
            mars_distance_km
        )


    # -----------------------------------------------------
    # 9.4 Final derived quantities
    # -----------------------------------------------------

    periapsis_altitude_km = (
        minimum_mars_distance_km
        - mars_radius_km
    )

    closest_approach_days = (
        closest_approach_time
        * 365.25
    )

    altitude_error_km = (
        periapsis_altitude_km
        - target_mars_altitude_km
    )


    return {
        "collision": collision,
        "phase_deg": mars_phase_deg,
        "periapsis_distance_km":
            minimum_mars_distance_km,
        "periapsis_altitude_km":
            periapsis_altitude_km,
        "altitude_error_km":
            altitude_error_km,
        "closest_approach_days":
            closest_approach_days,
        "relative_speed_km_s":
            relative_speed_at_periapsis_km_s,
        "earth_ship_distance_day5_km":
            earth_ship_distance_km,
        "earth_relative_speed_day5_km_s":
            earth_relative_speed_km_s,
        "escape_steps":
            escape_steps,
        "cruise_steps":
            cruise_steps,
        "encounter_steps":
            encounter_steps,
    }


# ---------------------------------------------------------
# 10. Run final validation
# ---------------------------------------------------------

print(
    "\n========================================="
)

print(
    "--- FINAL PHASE VALIDATION ---"
)

print(
    "Analytical Hohmann phase [deg]:",
    np.degrees(
        analytical_mars_phase
    ),
)

print(
    "Candidate N-body phase [deg]:",
    mars_phase_deg,
)

print(
    "Phase correction [deg]:",
    mars_phase_deg
    - np.degrees(
        analytical_mars_phase
    ),
)

print(
    "Target Mars altitude [km]:",
    target_mars_altitude_km,
)


result = run_final_mission()


# ---------------------------------------------------------
# 11. Display final result
# ---------------------------------------------------------

print(
    "\n========================================="
)

print(
    "--- FINAL RESULT ---"
)


if result["collision"]:

    print(
        "Status: COLLISION"
    )

else:

    print(
        "Status: SAFE FLYBY"
    )


print(
    "Mars phase [deg]:",
    result["phase_deg"],
)

print(
    "Mars periapsis centre distance [km]:",
    result["periapsis_distance_km"],
)

print(
    "Mars periapsis altitude [km]:",
    result["periapsis_altitude_km"],
)

print(
    "Target altitude [km]:",
    target_mars_altitude_km,
)

print(
    "Altitude error [km]:",
    result["altitude_error_km"],
)

print(
    "Closest approach time [days]:",
    result["closest_approach_days"],
)

print(
    "Relative speed at periapsis [km/s]:",
    result["relative_speed_km_s"],
)

print(
    "Earth-Ship distance after 5 days [km]:",
    result["earth_ship_distance_day5_km"],
)

print(
    "Earth-relative speed after 5 days [km/s]:",
    result["earth_relative_speed_day5_km_s"],
)


# ---------------------------------------------------------
# 12. Acceptance criterion
# ---------------------------------------------------------

if (
    not result["collision"]
    and abs(
        result["altitude_error_km"]
    )
    <= altitude_tolerance_km
):

    print(
        "\nFINAL PHASE ACCEPTED"
    )

    print(
        "The Mars flyby altitude is within",
        altitude_tolerance_km,
        "km of the target."
    )

else:

    print(
        "\nFINAL PHASE NOT YET ACCEPTED"
    )

    print(
        "A small final phase correction is required."
    )


# ---------------------------------------------------------
# 13. Numerical work summary
# ---------------------------------------------------------

print(
    "\n--- Integration summary ---"
)

print(
    "Earth escape steps:",
    result["escape_steps"],
)

print(
    "Cruise steps:",
    result["cruise_steps"],
)

print(
    "Mars encounter steps:",
    result["encounter_steps"],
)