import numpy as np

from bodies import Body

from physics import (
    G,
    circular_orbit_speed,
    hohmann_transfer_speed,
    hohmann_phase_angle,
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

mars_phase = hohmann_phase_angle(
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
# 6. Run the first five days of the mission
# ---------------------------------------------------------

def run_escape_test(target_dt):
    """
    Propagate the first five days of the mission.

    The final time is identical for every simulation,
    allowing a clean comparison between time steps.
    """

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
    # Spacecraft initial position
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
    # Spacecraft initial velocity
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


    bodies = [
        sun,
        earth,
        mars,
        ship,
    ]


    # -----------------------------------------------------
    # Five-day simulation
    # -----------------------------------------------------

    simulation_time = (
        5.0 / 365.25
    )

    number_of_steps = int(
        np.ceil(
            simulation_time
            / target_dt
        )
    )

    # Adjust dt slightly so every simulation ends at
    # exactly the same physical time.
    dt = (
        simulation_time
        / number_of_steps
    )


    for step in range(number_of_steps):

        velocity_verlet_step(
            bodies,
            dt,
            gravitational_acceleration,
        )


    # -----------------------------------------------------
    # Earth-relative state after exactly five days
    # -----------------------------------------------------

    relative_position = (
        ship.position
        - earth.position
    )

    relative_velocity = (
        ship.velocity
        - earth.velocity
    )


    earth_distance_km = (
        np.linalg.norm(
            relative_position
        )
        * AU_TO_KM
    )

    relative_speed_km_s = (
        np.linalg.norm(
            relative_velocity
        )
        * AU_TO_KM
        / YEAR_TO_SECONDS
    )

    velocity_direction_deg = np.degrees(
        np.arctan2(
            relative_velocity[1],
            relative_velocity[0],
        )
    )


    return (
        dt,
        number_of_steps,
        earth_distance_km,
        relative_speed_km_s,
        velocity_direction_deg,
        relative_position.copy(),
        relative_velocity.copy(),
    )


# ---------------------------------------------------------
# 7. Time steps to test
# ---------------------------------------------------------

time_steps = [
    1.0e-6,
    5.0e-7,
    2.5e-7,
    1.25e-7,
]


# ---------------------------------------------------------
# 8. Run convergence study
# ---------------------------------------------------------

print(
    "\n--- Earth escape convergence ---"
)

print(
    "State measured after exactly 5 days.\n"
)

print(
    f"{'dt [s]':>12} "
    f"{'Steps':>12} "
    f"{'Earth dist [km]':>18} "
    f"{'Rel speed [km/s]':>20} "
    f"{'Velocity angle [deg]':>22}"
)


results = []


for target_dt in time_steps:

    result = run_escape_test(
        target_dt
    )

    results.append(
        result
    )

    (
        actual_dt,
        number_of_steps,
        earth_distance_km,
        relative_speed_km_s,
        velocity_direction_deg,
        relative_position,
        relative_velocity,
    ) = result


    print(
        f"{actual_dt * YEAR_TO_SECONDS:12.4f} "
        f"{number_of_steps:12d} "
        f"{earth_distance_km:18.3f} "
        f"{relative_speed_km_s:20.8f} "
        f"{velocity_direction_deg:22.8f}"
    )


# ---------------------------------------------------------
# 9. Compare consecutive resolutions
# ---------------------------------------------------------

print(
    "\n--- Consecutive differences ---"
)


for i in range(
    len(results) - 1
):

    coarse = results[i]
    fine = results[i + 1]


    position_difference_km = (
        np.linalg.norm(
            coarse[5]
            - fine[5]
        )
        * AU_TO_KM
    )


    velocity_difference_km_s = (
        np.linalg.norm(
            coarse[6]
            - fine[6]
        )
        * AU_TO_KM
        / YEAR_TO_SECONDS
    )


    angle_difference_deg = abs(
        coarse[4]
        - fine[4]
    )


    print()

    print(
        "dt:",
        coarse[0] * YEAR_TO_SECONDS,
        "s ->",
        fine[0] * YEAR_TO_SECONDS,
        "s",
    )

    print(
        "Position difference [km]:",
        position_difference_km,
    )

    print(
        "Velocity difference [km/s]:",
        velocity_difference_km_s,
    )

    print(
        "Velocity angle difference [deg]:",
        angle_difference_deg,
    )