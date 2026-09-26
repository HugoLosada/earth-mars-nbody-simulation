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
# 5. TMI quantities
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
# 7. Run one simulation
# ---------------------------------------------------------

def run_transfer(
    mars_phase_deg,
    dt,
):
    """
    Run the N-body transfer for one Mars phase angle
    and one integration time step.

    Returns
    -------
    minimum_distance_km : float
        Minimum spacecraft-Mars distance.

    closest_approach_days : float
        Time at which minimum distance occurs.

    escape_speed_km_s : float
        Earth-relative spacecraft speed when it first
        reaches 1,000,000 km from Earth.

    escape_direction_deg : float
        Direction of the Earth-relative velocity at
        that point.
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
    # Simulation duration
    # -----------------------------------------------------

    simulation_time = (
        transfer_time
        + 30.0 / 365.25
    )

    number_of_steps = int(
        np.ceil(
            simulation_time / dt
        )
    )


    # -----------------------------------------------------
    # Quantities to measure
    # -----------------------------------------------------

    minimum_distance = np.inf
    closest_approach_time = 0.0

    escape_relative_velocity = None


    # -----------------------------------------------------
    # Numerical propagation
    # -----------------------------------------------------

    for step in range(number_of_steps):

        velocity_verlet_step(
            bodies,
            dt,
            gravitational_acceleration,
        )

        current_time = (
            (step + 1) * dt
        )


        # -------------------------------------------------
        # Mars distance
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Earth escape diagnostic
        # -------------------------------------------------

        earth_distance_km = (
            np.linalg.norm(
                ship.position
                - earth.position
            )
            * AU_TO_KM
        )

        if (
            escape_relative_velocity is None
            and earth_distance_km >= 1_000_000
        ):

            escape_relative_velocity = (
                ship.velocity
                - earth.velocity
            ).copy()


    # -----------------------------------------------------
    # Convert results
    # -----------------------------------------------------

    minimum_distance_km = (
        minimum_distance
        * AU_TO_KM
    )

    closest_approach_days = (
        closest_approach_time
        * 365.25
    )


    # -----------------------------------------------------
    # Escape diagnostics
    # -----------------------------------------------------

    if escape_relative_velocity is not None:

        escape_speed_km_s = (
            np.linalg.norm(
                escape_relative_velocity
            )
            * AU_TO_KM
            / YEAR_TO_SECONDS
        )

        escape_direction_deg = np.degrees(
            np.arctan2(
                escape_relative_velocity[1],
                escape_relative_velocity[0],
            )
        )

    else:

        escape_speed_km_s = np.nan
        escape_direction_deg = np.nan


    return (
        minimum_distance_km,
        closest_approach_days,
        escape_speed_km_s,
        escape_direction_deg,
    )


# ---------------------------------------------------------
# 8. Candidate phase
# ---------------------------------------------------------

mars_phase_deg = 45.1


# ---------------------------------------------------------
# 9. Time steps to compare
# ---------------------------------------------------------

time_steps = [
    5.0e-6,
    2.0e-6,
    1.0e-6,
]


# ---------------------------------------------------------
# 10. Run convergence comparison
# ---------------------------------------------------------

print(
    "\n--- Transfer time-step convergence ---"
)

print(
    "Mars phase [deg]:",
    mars_phase_deg,
)

print()

print(
    f"{'dt [years]':>12} "
    f"{'dt [s]':>12} "
    f"{'Min dist [km]':>18} "
    f"{'Time [days]':>15} "
    f"{'Escape v [km/s]':>18} "
    f"{'Escape angle [deg]':>20}"
)


results = []


for dt in time_steps:

    (
        minimum_distance_km,
        closest_approach_days,
        escape_speed_km_s,
        escape_direction_deg,
    ) = run_transfer(
        mars_phase_deg=mars_phase_deg,
        dt=dt,
    )

    dt_seconds = (
        dt * YEAR_TO_SECONDS
    )

    results.append(
        (
            dt,
            minimum_distance_km,
            closest_approach_days,
            escape_speed_km_s,
            escape_direction_deg,
        )
    )

    print(
        f"{dt:12.1e} "
        f"{dt_seconds:12.3f} "
        f"{minimum_distance_km:18.3f} "
        f"{closest_approach_days:15.3f} "
        f"{escape_speed_km_s:18.6f} "
        f"{escape_direction_deg:20.6f}"
    )


# ---------------------------------------------------------
# 11. Compare medium and high resolution
# ---------------------------------------------------------

medium_result = results[1]
high_result = results[2]

distance_difference = abs(
    medium_result[1]
    - high_result[1]
)

time_difference = abs(
    medium_result[2]
    - high_result[2]
)

escape_speed_difference = abs(
    medium_result[3]
    - high_result[3]
)

escape_angle_difference = abs(
    medium_result[4]
    - high_result[4]
)


print(
    "\n--- Difference: dt = 2e-6 vs dt = 1e-6 ---"
)

print(
    "Closest-approach distance difference [km]:",
    distance_difference,
)

print(
    "Closest-approach time difference [days]:",
    time_difference,
)

print(
    "Escape-speed difference [km/s]:",
    escape_speed_difference,
)

print(
    "Escape-angle difference [deg]:",
    escape_angle_difference,
)