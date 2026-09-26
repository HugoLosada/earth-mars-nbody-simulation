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

AU_PER_YEAR_TO_KM_PER_S = (
    AU_TO_KM / YEAR_TO_SECONDS
)


# ---------------------------------------------------------
# 2. Sun, Earth and Mars parameters
# ---------------------------------------------------------

sun_mass = 1.0

earth_mass = 3.004e-6
mars_mass = 3.214e-7

earth_orbit_radius = 1.0
mars_orbit_radius = 1.53


# ---------------------------------------------------------
# 3. Parking orbit around Earth
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
# 4. Heliocentric orbital quantities
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

transfer_time = hohmann_transfer_time(
    central_mass=sun_mass,
    departure_radius=earth_orbit_radius,
    arrival_radius=mars_orbit_radius,
)


# ---------------------------------------------------------
# 5. Hyperbolic excess velocity relative to Earth
# ---------------------------------------------------------

v_infinity = (
    hohmann_departure_speed
    - earth_speed
)


# ---------------------------------------------------------
# 6. Earth gravitational parameter
# ---------------------------------------------------------

mu_earth = (
    G * earth_mass
)


# ---------------------------------------------------------
# 7. Parking-orbit speed
# ---------------------------------------------------------

parking_speed = np.sqrt(
    mu_earth / parking_radius
)


# ---------------------------------------------------------
# 8. Hyperbolic injection speed at periapsis
# ---------------------------------------------------------

injection_speed = hyperbolic_periapsis_speed(
    mu=mu_earth,
    periapsis_radius=parking_radius,
    v_infinity=v_infinity,
)


# ---------------------------------------------------------
# 9. TMI delta-v
# ---------------------------------------------------------

delta_v_tmi = (
    injection_speed
    - parking_speed
)


# ---------------------------------------------------------
# 10. Hyperbolic escape geometry
# ---------------------------------------------------------

hyperbolic_e = hyperbolic_eccentricity(
    mu=mu_earth,
    periapsis_radius=parking_radius,
    v_infinity=v_infinity,
)

asymptote_true_anomaly = hyperbolic_asymptote_angle(
    hyperbolic_e
)


# Desired asymptotic v-infinity direction:
# parallel to Earth's prograde heliocentric velocity (+y).
desired_v_infinity_angle = np.pi / 2


# For the outbound branch:
#
# periapsis_angle + asymptote_true_anomaly
# = desired asymptotic direction
#
periapsis_angle = (
    desired_v_infinity_angle
    - asymptote_true_anomaly
)


# ---------------------------------------------------------
# 11. Create Sun
# ---------------------------------------------------------

sun = Body(
    name="Sun",
    mass=sun_mass,
    position=(0.0, 0.0),
    velocity=(0.0, 0.0),
)


# ---------------------------------------------------------
# 12. Create Earth
# ---------------------------------------------------------

earth = Body(
    name="Earth",
    mass=earth_mass,
    position=(earth_orbit_radius, 0.0),
    velocity=(0.0, earth_speed),
)


# ---------------------------------------------------------
# 13. Create Mars
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 14. Spacecraft position at hyperbolic periapsis
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 15. Spacecraft velocity at hyperbolic periapsis
# ---------------------------------------------------------

# For counter-clockwise motion, the tangential unit vector
# perpendicular to the radius vector is:
#
# (-sin(theta), cos(theta))
#

ship_relative_velocity = (
    injection_speed
    * np.array(
        [
            -np.sin(periapsis_angle),
            np.cos(periapsis_angle),
        ]
    )
)


# Convert Earth-relative velocity into heliocentric velocity.
ship_velocity = (
    earth.velocity
    + ship_relative_velocity
)


# ---------------------------------------------------------
# 16. Create spacecraft
# ---------------------------------------------------------

ship = Body(
    name="Ship",
    mass=0.0,
    position=ship_position,
    velocity=ship_velocity,
)


# ---------------------------------------------------------
# 17. N-body system
# ---------------------------------------------------------

bodies = [
    sun,
    earth,
    mars,
    ship,
]


# ---------------------------------------------------------
# 18. Display mission and escape parameters
# ---------------------------------------------------------

print("\n--- Mission parameters ---")

print(
    "Parking orbit altitude [km]:",
    parking_altitude_km,
)

print(
    "Parking orbit speed [km/s]:",
    parking_speed
    * AU_PER_YEAR_TO_KM_PER_S,
)

print(
    "Hyperbolic excess velocity [km/s]:",
    v_infinity
    * AU_PER_YEAR_TO_KM_PER_S,
)

print(
    "Injection speed [km/s]:",
    injection_speed
    * AU_PER_YEAR_TO_KM_PER_S,
)

print(
    "TMI delta-v [km/s]:",
    delta_v_tmi
    * AU_PER_YEAR_TO_KM_PER_S,
)

print(
    "Ideal Hohmann transfer time [days]:",
    transfer_time * 365.25,
)

print(
    "Initial Mars phase angle [deg]:",
    np.degrees(mars_phase),
)


print("\n--- Hyperbolic escape geometry ---")

print(
    "Hyperbolic eccentricity:",
    hyperbolic_e,
)

print(
    "Asymptote true anomaly [deg]:",
    np.degrees(
        asymptote_true_anomaly
    ),
)

print(
    "Periapsis orientation [deg]:",
    np.degrees(
        periapsis_angle
    ),
)


# ---------------------------------------------------------
# 19. Initial-condition checks
# ---------------------------------------------------------

initial_earth_ship_distance = np.linalg.norm(
    ship.position - earth.position
)

initial_ship_relative_speed = np.linalg.norm(
    ship.velocity - earth.velocity
)

print("\n--- Initial conditions ---")

print(
    "Earth-Ship distance [km]:",
    initial_earth_ship_distance
    * AU_TO_KM,
)

print(
    "Ship heliocentric speed [km/s]:",
    np.linalg.norm(
        ship.velocity
    )
    * AU_PER_YEAR_TO_KM_PER_S,
)

print(
    "Ship speed relative to Earth [km/s]:",
    initial_ship_relative_speed
    * AU_PER_YEAR_TO_KM_PER_S,
)


# ---------------------------------------------------------
# 20. Numerical simulation parameters
# ---------------------------------------------------------

dt = 1.0e-6

extra_time_days = 30.0

simulation_time = (
    transfer_time
    + extra_time_days / 365.25
)

number_of_steps = int(
    np.ceil(
        simulation_time / dt
    )
)

print("\n--- Numerical simulation ---")

print(
    "Time step [seconds]:",
    dt * YEAR_TO_SECONDS,
)

print(
    "Number of integration steps:",
    number_of_steps,
)

print(
    "Simulated duration [days]:",
    number_of_steps
    * dt
    * 365.25,
)


# ---------------------------------------------------------
# 21. Quantities monitored during propagation
# ---------------------------------------------------------

minimum_mars_distance = np.inf
minimum_mars_distance_time = 0.0

maximum_earth_distance = 0.0

# Diagnostic value:
# spacecraft velocity relative to Earth once it is far away.
escape_relative_velocity = None


# ---------------------------------------------------------
# 22. Propagate full N-body system
# ---------------------------------------------------------

for step in range(number_of_steps):

    velocity_verlet_step(
        bodies,
        dt,
        gravitational_acceleration,
    )

    current_time = (
        (step + 1) * dt
    )

    earth_ship_vector = (
        ship.position
        - earth.position
    )

    earth_ship_distance = np.linalg.norm(
        earth_ship_vector
    )

    if earth_ship_distance > maximum_earth_distance:
        maximum_earth_distance = (
            earth_ship_distance
        )

    # Diagnostic:
    # once the ship is more than 1 million km from Earth,
    # record its Earth-relative velocity.
    if (
        escape_relative_velocity is None
        and earth_ship_distance
        * AU_TO_KM
        > 1_000_000
    ):

        escape_relative_velocity = (
            ship.velocity
            - earth.velocity
        ).copy()

    mars_ship_distance = np.linalg.norm(
        ship.position
        - mars.position
    )

    if mars_ship_distance < minimum_mars_distance:

        minimum_mars_distance = (
            mars_ship_distance
        )

        minimum_mars_distance_time = (
            current_time
        )


# ---------------------------------------------------------
# 23. Convert results
# ---------------------------------------------------------

minimum_mars_distance_km = (
    minimum_mars_distance
    * AU_TO_KM
)

closest_approach_days = (
    minimum_mars_distance_time
    * 365.25
)

final_earth_ship_distance_km = (
    np.linalg.norm(
        ship.position
        - earth.position
    )
    * AU_TO_KM
)


# ---------------------------------------------------------
# 24. Display N-body results
# ---------------------------------------------------------

print("\n--- N-body results ---")

print(
    "Minimum Ship-Mars distance [km]:",
    minimum_mars_distance_km,
)

print(
    "Time of closest approach [days]:",
    closest_approach_days,
)

print(
    "Ideal Hohmann arrival time [days]:",
    transfer_time * 365.25,
)

print(
    "Difference from ideal arrival time [days]:",
    closest_approach_days
    - transfer_time * 365.25,
)

print(
    "Final Earth-Ship distance [km]:",
    final_earth_ship_distance_km,
)


# ---------------------------------------------------------
# 25. Check asymptotic escape direction
# ---------------------------------------------------------

if escape_relative_velocity is not None:

    escape_speed_km_s = (
        np.linalg.norm(
            escape_relative_velocity
        )
        * AU_PER_YEAR_TO_KM_PER_S
    )

    escape_direction_deg = np.degrees(
        np.arctan2(
            escape_relative_velocity[1],
            escape_relative_velocity[0],
        )
    )

    print(
        "Earth-relative escape speed "
        "at 1,000,000 km [km/s]:",
        escape_speed_km_s,
    )

    print(
        "Earth-relative escape direction [deg]:",
        escape_direction_deg,
    )