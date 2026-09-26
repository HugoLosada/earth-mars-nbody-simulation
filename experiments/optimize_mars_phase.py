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
# 7. Run one N-body transfer for a chosen Mars phase
# ---------------------------------------------------------

def run_transfer(
    mars_phase_deg,
    dt=5.0e-6,
    extra_time_days=30.0,
):
    """
    Run one complete N-body Earth-Mars transfer.

    Parameters
    ----------
    mars_phase_deg : float
        Initial Mars phase angle in degrees.

    dt : float
        Integration time step in years.

    extra_time_days : float
        Additional propagation time beyond the analytical
        Hohmann transfer time.

    Returns
    -------
    tuple
        Minimum Ship-Mars distance in km and
        time of closest approach in days.
    """

    mars_phase = np.radians(
        mars_phase_deg
    )

    # -----------------------------------------------------
    # Create Sun
    # -----------------------------------------------------

    sun = Body(
        name="Sun",
        mass=sun_mass,
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
    )

    # -----------------------------------------------------
    # Create Earth
    # -----------------------------------------------------

    earth = Body(
        name="Earth",
        mass=earth_mass,
        position=(earth_orbit_radius, 0.0),
        velocity=(0.0, earth_speed),
    )

    # -----------------------------------------------------
    # Create Mars using the phase being tested
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
    # Spacecraft initial Earth-relative position
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
    # Spacecraft initial Earth-relative velocity
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
    # Create spacecraft
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
        + extra_time_days / 365.25
    )

    number_of_steps = int(
        np.ceil(
            simulation_time / dt
        )
    )

    # -----------------------------------------------------
    # Search for closest approach
    # -----------------------------------------------------

    minimum_distance = np.inf
    closest_approach_time = 0.0

    for step in range(number_of_steps):

        velocity_verlet_step(
            bodies,
            dt,
            gravitational_acceleration,
        )

        ship_mars_distance = np.linalg.norm(
            ship.position
            - mars.position
        )

        if ship_mars_distance < minimum_distance:

            minimum_distance = (
                ship_mars_distance
            )

            closest_approach_time = (
                (step + 1) * dt
            )

    minimum_distance_km = (
        minimum_distance
        * AU_TO_KM
    )

    closest_approach_days = (
        closest_approach_time
        * 365.25
    )

    return (
        minimum_distance_km,
        closest_approach_days,
    )


# ---------------------------------------------------------
# 8. Display analytical reference
# ---------------------------------------------------------

print(
    "Analytical Hohmann Mars phase [deg]:",
    np.degrees(
        analytical_mars_phase
    ),
)

print(
    "Analytical Hohmann transfer time [days]:",
    transfer_time * 365.25,
)


# ---------------------------------------------------------
# 9. Coarse phase-angle search
# ---------------------------------------------------------

print(
    "\n--- Coarse phase search ---"
)

coarse_phases = np.arange(
    40.0,
    50.1,
    1.0,
)

coarse_results = []

print(
    f"{'Phase [deg]':>15} "
    f"{'Min distance [km]':>20} "
    f"{'Closest approach [days]':>25}"
)


for phase in coarse_phases:

    (
        minimum_distance_km,
        closest_approach_days,
    ) = run_transfer(
        mars_phase_deg=phase,
        dt=5.0e-6,
    )

    coarse_results.append(
        (
            phase,
            minimum_distance_km,
            closest_approach_days,
        )
    )

    print(
        f"{phase:15.3f} "
        f"{minimum_distance_km:20.3f} "
        f"{closest_approach_days:25.3f}"
    )


# ---------------------------------------------------------
# 10. Find best coarse result
# ---------------------------------------------------------

best_coarse_result = min(
    coarse_results,
    key=lambda result: result[1],
)

best_coarse_phase = (
    best_coarse_result[0]
)

print(
    "\nBest coarse phase [deg]:",
    best_coarse_phase,
)

print(
    "Best coarse distance [km]:",
    best_coarse_result[1],
)


# ---------------------------------------------------------
# 11. Fine search around the best coarse phase
# ---------------------------------------------------------

print(
    "\n--- Fine phase search ---"
)

fine_phases = np.arange(
    best_coarse_phase - 0.5,
    best_coarse_phase + 0.5001,
    0.1,
)

fine_results = []


for phase in fine_phases:

    (
        minimum_distance_km,
        closest_approach_days,
    ) = run_transfer(
        mars_phase_deg=phase,
        dt=5.0e-6,
    )

    fine_results.append(
        (
            phase,
            minimum_distance_km,
            closest_approach_days,
        )
    )

    print(
        f"{phase:15.3f} "
        f"{minimum_distance_km:20.3f} "
        f"{closest_approach_days:25.3f}"
    )


# ---------------------------------------------------------
# 12. Best phase found
# ---------------------------------------------------------

best_result = min(
    fine_results,
    key=lambda result: result[1],
)

best_phase = best_result[0]
best_distance = best_result[1]
best_time = best_result[2]

# ---------------------------------------------------------
# 13. Very fine search around the best phase
# ---------------------------------------------------------

print(
    "\n--- Very fine phase search ---"
)

very_fine_phases = np.arange(
    best_phase - 0.1,
    best_phase + 0.1001,
    0.01,
)

very_fine_results = []

for phase in very_fine_phases:

    (
        minimum_distance_km,
        closest_approach_days,
    ) = run_transfer(
        mars_phase_deg=phase,
        dt=2.0e-6,
    )

    very_fine_results.append(
        (
            phase,
            minimum_distance_km,
            closest_approach_days,
        )
    )

    print(
        f"{phase:15.3f} "
        f"{minimum_distance_km:20.3f} "
        f"{closest_approach_days:25.3f}"
    )


best_very_fine_result = min(
    very_fine_results,
    key=lambda result: result[1],
)

best_very_fine_phase = (
    best_very_fine_result[0]
)


print(
    "\n--- Best very fine result ---"
)

print(
    "Optimal Mars phase [deg]:",
    best_very_fine_phase,
)

print(
    "Minimum Ship-Mars distance [km]:",
    best_very_fine_result[1],
)

print(
    "Closest approach time [days]:",
    best_very_fine_result[2],
)


print(
    "\n--- Best N-body phase ---"
)

print(
    "Optimal Mars phase [deg]:",
    best_phase,
)

print(
    "Minimum Ship-Mars distance [km]:",
    best_distance,
)

print(
    "Closest approach time [days]:",
    best_time,
)

print(
    "Difference from analytical Hohmann phase [deg]:",
    best_phase
    - np.degrees(
        analytical_mars_phase
    ),
)