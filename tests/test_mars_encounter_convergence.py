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


# Mars phase found in the previous refinement.
mars_phase_deg = 45.19695312500001


# ---------------------------------------------------------
# 4. Heliocentric quantities
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
# 5. Earth escape parameters
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

# ~3.94 seconds
escape_dt_target = (
    1.25e-7
)

# ~63.1 seconds
cruise_dt_target = (
    2.0e-6
)

# Start the close-Mars integration here.
mars_close_region_km = (
    1_000_000.0
)

simulation_end_time = (
    transfer_time
    + 30.0 / 365.25
)


# ---------------------------------------------------------
# 7. Create initial system
# ---------------------------------------------------------

def create_system():
    """
    Create the initial Sun-Earth-Mars-spacecraft system.
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
# 8. Clone system
# ---------------------------------------------------------

def clone_system(bodies):
    """
    Create an independent copy of the complete system.
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
# 9. Propagate once to the Mars close-encounter region
# ---------------------------------------------------------

def propagate_to_mars_region(bodies):
    """
    Propagate the mission until the spacecraft first
    enters the 1,000,000 km region around Mars.

    Earth escape uses the validated fine step.
    Interplanetary cruise uses the validated coarse step.
    """

    mars = bodies[2]
    ship = bodies[3]

    current_time = 0.0


    # -----------------------------------------------------
    # 9.1 High-resolution Earth escape
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
        "\n--- Earth escape ---",
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

        current_time += (
            escape_dt
        )


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
                f"Escape progress: "
                f"{progress:.1f}%",
                flush=True,
            )


    # -----------------------------------------------------
    # 9.2 Interplanetary cruise
    # -----------------------------------------------------

    print(
        "\n--- Cruise to Mars encounter region ---",
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


        # Stop BEFORE doing another cruise step once
        # the spacecraft has entered the close region.
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
            cruise_steps % 100_000
            == 0
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


    mars_distance_km = (
        np.linalg.norm(
            ship.position
            - mars.position
        )
        * AU_TO_KM
    )


    print(
        "\nMars close region reached.",
        flush=True,
    )

    print(
        "Mission time [days]:",
        current_time * 365.25,
        flush=True,
    )

    print(
        "Ship-Mars distance [km]:",
        mars_distance_km,
        flush=True,
    )

    print(
        "Cruise steps:",
        cruise_steps,
        flush=True,
    )


    return (
        bodies,
        current_time,
    )


# ---------------------------------------------------------
# 10. Integrate one close encounter
# ---------------------------------------------------------

def run_close_encounter(
    initial_bodies,
    initial_time,
    target_dt,
):
    """
    Starting from the same 1,000,000 km Mars encounter
    state, propagate through periapsis using one chosen
    time step.
    """

    bodies = clone_system(
        initial_bodies
    )

    mars = bodies[2]
    ship = bodies[3]

    current_time = (
        initial_time
    )


    minimum_distance_km = (
        np.linalg.norm(
            ship.position
            - mars.position
        )
        * AU_TO_KM
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


    previous_distance_km = (
        minimum_distance_km
    )

    number_of_steps = 0

    collision = False


    # -----------------------------------------------------
    # Integrate through periapsis
    # -----------------------------------------------------

    while current_time < simulation_end_time:

        dt = (
            target_dt
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

        number_of_steps += 1


        mars_distance_km = (
            np.linalg.norm(
                ship.position
                - mars.position
            )
            * AU_TO_KM
        )


        # -------------------------------------------------
        # Collision check
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
        # Record new minimum
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
        # Once distance starts increasing, periapsis
        # has been passed.
        # -------------------------------------------------

        if (
            mars_distance_km
            > previous_distance_km
            and previous_distance_km
            <= minimum_distance_km
        ):

            break


        previous_distance_km = (
            mars_distance_km
        )


    periapsis_altitude_km = (
        minimum_distance_km
        - mars_radius_km
    )

    closest_approach_days = (
        closest_approach_time
        * 365.25
    )


    return {
        "dt": target_dt,
        "steps": number_of_steps,
        "collision": collision,
        "distance_km": minimum_distance_km,
        "altitude_km": periapsis_altitude_km,
        "time_days": closest_approach_days,
        "relative_speed_km_s": relative_speed_km_s,
    }


# ---------------------------------------------------------
# 11. Build common pre-encounter state
# ---------------------------------------------------------

print(
    "\n========================================="
)

print(
    "--- Mars encounter convergence test ---"
)

print(
    "Mars phase [deg]:",
    mars_phase_deg,
)

print(
    "Mars radius [km]:",
    mars_radius_km,
)


initial_bodies = (
    create_system()
)


(
    encounter_bodies,
    encounter_start_time,
) = propagate_to_mars_region(
    initial_bodies
)


# ---------------------------------------------------------
# 12. Close-encounter time steps
# ---------------------------------------------------------

close_time_steps = [
    1.25e-7,
    6.25e-8,
    3.125e-8,
]


# ---------------------------------------------------------
# 13. Run convergence comparison
# ---------------------------------------------------------

results = []


print(
    "\n--- Close-Mars simulations ---"
)


for simulation_number, dt in enumerate(
    close_time_steps,
    start=1,
):

    print(
        "\n-----------------------------------------"
    )

    print(
        f"Simulation "
        f"{simulation_number}/"
        f"{len(close_time_steps)}"
    )

    print(
        "Close-Mars dt [s]:",
        dt * YEAR_TO_SECONDS,
    )


    result = run_close_encounter(
        initial_bodies=encounter_bodies,
        initial_time=encounter_start_time,
        target_dt=dt,
    )


    results.append(
        result
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
        "Integration steps:",
        result["steps"],
    )

    print(
        "Periapsis altitude [km]:",
        result["altitude_km"],
    )

    print(
        "Closest approach [days]:",
        result["time_days"],
    )

    print(
        "Relative speed [km/s]:",
        result["relative_speed_km_s"],
    )


# ---------------------------------------------------------
# 14. Summary table
# ---------------------------------------------------------

print(
    "\n--- Mars encounter convergence results ---"
)

print(
    f"{'dt [s]':>12} "
    f"{'Steps':>12} "
    f"{'Status':>14} "
    f"{'Altitude [km]':>18} "
    f"{'Time [days]':>16} "
    f"{'Rel speed [km/s]':>20}"
)


for result in results:

    status = (
        "COLLISION"
        if result["collision"]
        else "SAFE"
    )

    print(
        f'{result["dt"] * YEAR_TO_SECONDS:12.4f} '
        f'{result["steps"]:12d} '
        f'{status:>14} '
        f'{result["altitude_km"]:18.6f} '
        f'{result["time_days"]:16.9f} '
        f'{result["relative_speed_km_s"]:20.9f}'
    )


# ---------------------------------------------------------
# 15. Consecutive differences
# ---------------------------------------------------------

print(
    "\n--- Consecutive differences ---"
)


for i in range(
    len(results) - 1
):

    coarse = results[i]
    fine = results[i + 1]

    print()

    print(
        "dt:",
        coarse["dt"]
        * YEAR_TO_SECONDS,
        "s ->",
        fine["dt"]
        * YEAR_TO_SECONDS,
        "s",
    )


    if (
        coarse["collision"]
        or fine["collision"]
    ):

        print(
            "Cannot compare normally because "
            "one trajectory collided with Mars."
        )

        continue


    altitude_difference = abs(
        coarse["altitude_km"]
        - fine["altitude_km"]
    )

    time_difference_seconds = (
        abs(
            coarse["time_days"]
            - fine["time_days"]
        )
        * 24
        * 3600
    )

    speed_difference = abs(
        coarse["relative_speed_km_s"]
        - fine["relative_speed_km_s"]
    )


    print(
        "Periapsis altitude difference [km]:",
        altitude_difference,
    )

    print(
        "Closest-approach time difference [s]:",
        time_difference_seconds,
    )

    print(
        "Relative-speed difference [km/s]:",
        speed_difference,
    )


# ---------------------------------------------------------
# 16. Estimate convergence order
# ---------------------------------------------------------

if (
    len(results) == 3
    and not results[0]["collision"]
    and not results[1]["collision"]
    and not results[2]["collision"]
):

    difference_1 = abs(
        results[0]["altitude_km"]
        - results[1]["altitude_km"]
    )

    difference_2 = abs(
        results[1]["altitude_km"]
        - results[2]["altitude_km"]
    )


    if (
        difference_1 > 0
        and difference_2 > 0
    ):

        observed_order = (
            np.log(
                difference_1
                / difference_2
            )
            / np.log(2)
        )

        print(
            "\n--- Observed convergence ---"
        )

        print(
            "Estimated order from periapsis altitude:",
            observed_order,
        )