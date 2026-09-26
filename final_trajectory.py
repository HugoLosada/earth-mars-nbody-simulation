import csv
from pathlib import Path

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


# =========================================================
# 1. UNIT CONVERSIONS
# =========================================================

AU_TO_KM = 149_597_870.7

YEAR_TO_SECONDS = (
    365.25 * 24 * 3600
)


# =========================================================
# 2. PHYSICAL PARAMETERS
# =========================================================

sun_mass = 1.0

earth_mass = 3.004e-6
mars_mass = 3.214e-7

earth_orbit_radius = 1.0
mars_orbit_radius = 1.53

earth_radius_km = 6371.0
mars_radius_km = 3389.5


# =========================================================
# 3. FINAL MISSION PARAMETERS
# =========================================================

parking_altitude_km = 300.0

parking_radius_km = (
    earth_radius_km
    + parking_altitude_km
)

parking_radius = (
    parking_radius_km
    / AU_TO_KM
)


# Final phase obtained from the validated optimization.
mars_phase_deg = 45.1969725


# Target Mars flyby altitude.
target_mars_altitude_km = 300.0


# =========================================================
# 4. ANALYTICAL HOHMANN REFERENCE
# =========================================================

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


# =========================================================
# 5. EARTH ESCAPE PARAMETERS
# =========================================================

v_infinity = (
    hohmann_departure_speed
    - earth_speed
)

mu_earth = (
    G * earth_mass
)

parking_speed = np.sqrt(
    mu_earth
    / parking_radius
)

injection_speed = hyperbolic_periapsis_speed(
    mu=mu_earth,
    periapsis_radius=parking_radius,
    v_infinity=v_infinity,
)

delta_v_tmi = (
    injection_speed
    - parking_speed
)


# =========================================================
# 6. EARTH HYPERBOLIC ESCAPE GEOMETRY
# =========================================================

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


# =========================================================
# 7. VALIDATED NUMERICAL PARAMETERS
# =========================================================

escape_duration_days = 5.0


# Approximately 3.94 seconds.
escape_dt_target = (
    1.25e-7
)


# Approximately 63.1 seconds.
cruise_dt_target = (
    2.0e-6
)


# Approximately 3.94 seconds.
mars_close_dt_target = (
    1.25e-7
)


# Start high-resolution Mars integration here.
mars_close_region_km = (
    1_000_000.0
)


# Continue some distance beyond periapsis so the
# outbound part of the flyby can also be plotted.
mars_post_encounter_distance_km = (
    200_000.0
)


# =========================================================
# 8. TRAJECTORY SAMPLING
# =========================================================

# We do NOT save every integration step.

# Earth escape:
# one sample approximately every 2 minutes.
earth_save_every = 30


# Cruise:
# one sample approximately every 8.8 hours.
cruise_save_every = 500


# Mars encounter:
# one sample approximately every 39 seconds.
mars_save_every = 10


# =========================================================
# 9. OUTPUT FILE
# =========================================================

output_directory = Path(__file__).resolve().parent / "results"

output_directory.mkdir(
    exist_ok=True
)

output_file = (
    output_directory
    / "final_trajectory.csv"
)


# =========================================================
# 10. CREATE INITIAL SYSTEM
# =========================================================

def create_system():
    """
    Create the final optimized mission configuration.
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
    # Spacecraft initial Earth-relative position
    # -----------------------------------------------------

    ship_relative_position = np.array(
        [
            parking_radius
            * np.cos(
                periapsis_angle
            ),

            parking_radius
            * np.sin(
                periapsis_angle
            ),
        ]
    )


    ship_position = (
        earth.position
        + ship_relative_position
    )


    # -----------------------------------------------------
    # Spacecraft injection velocity relative to Earth
    # -----------------------------------------------------

    ship_relative_velocity = (
        injection_speed
        * np.array(
            [
                -np.sin(
                    periapsis_angle
                ),

                np.cos(
                    periapsis_angle
                ),
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


# =========================================================
# 11. RECORD ONE SYSTEM STATE
# =========================================================

def record_state(
    records,
    bodies,
    current_time,
    phase,
    is_periapsis=False,
):
    """
    Store one sampled state of the mission.

    Both heliocentric coordinates and planet-relative
    coordinates are saved so that later plots do not
    need to rerun the simulation.
    """

    sun = bodies[0]
    earth = bodies[1]
    mars = bodies[2]
    ship = bodies[3]


    earth_relative_position = (
        ship.position
        - earth.position
    )

    mars_relative_position = (
        ship.position
        - mars.position
    )


    earth_relative_velocity = (
        ship.velocity
        - earth.velocity
    )

    mars_relative_velocity = (
        ship.velocity
        - mars.velocity
    )


    record = {
        "time_days":
            current_time * 365.25,

        "phase":
            phase,

        "is_periapsis":
            int(is_periapsis),

        "sun_x_au":
            sun.position[0],

        "sun_y_au":
            sun.position[1],

        "earth_x_au":
            earth.position[0],

        "earth_y_au":
            earth.position[1],

        "mars_x_au":
            mars.position[0],

        "mars_y_au":
            mars.position[1],

        "ship_x_au":
            ship.position[0],

        "ship_y_au":
            ship.position[1],

        "ship_earth_x_km":
            earth_relative_position[0]
            * AU_TO_KM,

        "ship_earth_y_km":
            earth_relative_position[1]
            * AU_TO_KM,

        "earth_ship_distance_km":
            np.linalg.norm(
                earth_relative_position
            )
            * AU_TO_KM,

        "earth_relative_speed_km_s":
            np.linalg.norm(
                earth_relative_velocity
            )
            * AU_TO_KM
            / YEAR_TO_SECONDS,

        "ship_mars_x_km":
            mars_relative_position[0]
            * AU_TO_KM,

        "ship_mars_y_km":
            mars_relative_position[1]
            * AU_TO_KM,

        "mars_ship_distance_km":
            np.linalg.norm(
                mars_relative_position
            )
            * AU_TO_KM,

        "mars_relative_speed_km_s":
            np.linalg.norm(
                mars_relative_velocity
            )
            * AU_TO_KM
            / YEAR_TO_SECONDS,
    }


    records.append(
        record
    )


# =========================================================
# 12. RUN FINAL TRAJECTORY
# =========================================================

def run_final_trajectory():

    bodies = create_system()

    earth = bodies[1]
    mars = bodies[2]
    ship = bodies[3]


    records = []

    current_time = 0.0


    # Save initial state.
    record_state(
        records=records,
        bodies=bodies,
        current_time=current_time,
        phase="initial",
    )


    # -----------------------------------------------------
    # 12.1 EARTH ESCAPE
    # -----------------------------------------------------

    print(
        "\n--- Phase 1: Earth escape ---",
        flush=True,
    )


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
        "dt [s]:",
        escape_dt
        * YEAR_TO_SECONDS,
        flush=True,
    )

    print(
        "Steps:",
        escape_steps,
        flush=True,
    )


    for step in range(
        escape_steps
    ):

        velocity_verlet_step(
            bodies,
            escape_dt,
            gravitational_acceleration,
        )

        current_time += (
            escape_dt
        )


        if (
            (step + 1)
            % earth_save_every
            == 0

            or

            step + 1
            == escape_steps
        ):

            record_state(
                records=records,
                bodies=bodies,
                current_time=current_time,
                phase="earth_escape",
            )


        if (
            (step + 1)
            % 25_000
            == 0

            or

            step + 1
            == escape_steps
        ):

            progress = (
                100
                * (step + 1)
                / escape_steps
            )

            print(
                f"Progress: "
                f"{progress:.1f}%",
                flush=True,
            )


    # -----------------------------------------------------
    # State after five days
    # -----------------------------------------------------

    earth_distance_day5_km = (
        np.linalg.norm(
            ship.position
            - earth.position
        )
        * AU_TO_KM
    )


    earth_speed_day5_km_s = (
        np.linalg.norm(
            ship.velocity
            - earth.velocity
        )
        * AU_TO_KM
        / YEAR_TO_SECONDS
    )


    # -----------------------------------------------------
    # 12.2 INTERPLANETARY CRUISE
    # -----------------------------------------------------

    print(
        "\n--- Phase 2: Interplanetary cruise ---",
        flush=True,
    )


    cruise_steps = 0


    while True:

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


        velocity_verlet_step(
            bodies,
            cruise_dt_target,
            gravitational_acceleration,
        )

        current_time += (
            cruise_dt_target
        )

        cruise_steps += 1


        if (
            cruise_steps
            % cruise_save_every
            == 0
        ):

            record_state(
                records=records,
                bodies=bodies,
                current_time=current_time,
                phase="cruise",
            )


        if (
            cruise_steps
            % 100_000
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


    # Save exact transition into encounter region.
    record_state(
        records=records,
        bodies=bodies,
        current_time=current_time,
        phase="mars_encounter",
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
    # 12.3 CLOSE MARS ENCOUNTER
    # -----------------------------------------------------

    print(
        "\n--- Phase 3: Mars encounter ---",
        flush=True,
    )


    print(
        "dt [s]:",
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


    minimum_mars_distance_km = (
        previous_mars_distance_km
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


    periapsis_record = None

    passed_periapsis = False
    collision = False

    encounter_steps = 0


    while True:

        velocity_verlet_step(
            bodies,
            mars_close_dt_target,
            gravitational_acceleration,
        )

        current_time += (
            mars_close_dt_target
        )

        encounter_steps += 1


        mars_distance_km = (
            np.linalg.norm(
                ship.position
                - mars.position
            )
            * AU_TO_KM
        )


        # -------------------------------------------------
        # Save regular encounter samples
        # -------------------------------------------------

        if (
            encounter_steps
            % mars_save_every
            == 0
        ):

            record_state(
                records=records,
                bodies=bodies,
                current_time=current_time,
                phase="mars_encounter",
            )


        # -------------------------------------------------
        # Collision detection
        # -------------------------------------------------

        if (
            mars_distance_km
            <= mars_radius_km
        ):

            collision = True

            print(
                "\nCollision detected.",
                flush=True,
            )

            break


        # -------------------------------------------------
        # New closest approach
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


            # Store the exact current state separately.
            temporary_records = []

            record_state(
                records=temporary_records,
                bodies=bodies,
                current_time=current_time,
                phase="periapsis",
                is_periapsis=True,
            )

            periapsis_record = (
                temporary_records[0]
            )


        # -------------------------------------------------
        # Detect passage through periapsis
        # -------------------------------------------------

        if (
            mars_distance_km
            > previous_mars_distance_km
        ):

            passed_periapsis = True


        previous_mars_distance_km = (
            mars_distance_km
        )


        # -------------------------------------------------
        # Continue after periapsis to show outbound flyby
        # -------------------------------------------------

        if (
            passed_periapsis
            and mars_distance_km
            >= mars_post_encounter_distance_km
        ):

            record_state(
                records=records,
                bodies=bodies,
                current_time=current_time,
                phase="mars_departure",
            )

            break


    # -----------------------------------------------------
    # Add exact periapsis state
    # -----------------------------------------------------

    if (
        periapsis_record
        is not None
    ):

        records.append(
            periapsis_record
        )


    # Sort because the exact periapsis record was
    # appended after completing the encounter.
    records.sort(
        key=lambda record:
            record["time_days"]
    )


    # -----------------------------------------------------
    # Derived final results
    # -----------------------------------------------------

    periapsis_altitude_km = (
        minimum_mars_distance_km
        - mars_radius_km
    )


    closest_approach_days = (
        closest_approach_time
        * 365.25
    )


    return {
        "records":
            records,

        "collision":
            collision,

        "periapsis_distance_km":
            minimum_mars_distance_km,

        "periapsis_altitude_km":
            periapsis_altitude_km,

        "closest_approach_days":
            closest_approach_days,

        "relative_speed_km_s":
            relative_speed_at_periapsis_km_s,

        "earth_distance_day5_km":
            earth_distance_day5_km,

        "earth_speed_day5_km_s":
            earth_speed_day5_km_s,

        "escape_steps":
            escape_steps,

        "cruise_steps":
            cruise_steps,

        "encounter_steps":
            encounter_steps,
    }


# =========================================================
# 13. SAVE CSV
# =========================================================

def save_trajectory(records):
    """
    Write all sampled trajectory states to CSV.
    """

    if len(records) == 0:

        raise ValueError(
            "No trajectory records were generated."
        )


    fieldnames = list(
        records[0].keys()
    )


    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# =========================================================
# 14. RUN FINAL MISSION
# =========================================================

print(
    "\n========================================="
)

print(
    "--- FINAL TRAJECTORY GENERATION ---"
)

print(
    "Analytical Hohmann phase [deg]:",
    np.degrees(
        analytical_mars_phase
    ),
)

print(
    "Optimized Mars phase [deg]:",
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
    "TMI delta-v [km/s]:",
    delta_v_tmi
    * AU_TO_KM
    / YEAR_TO_SECONDS,
)


result = run_final_trajectory()


# =========================================================
# 15. SAVE TRAJECTORY
# =========================================================

save_trajectory(
    result["records"]
)


# =========================================================
# 16. DISPLAY FINAL SUMMARY
# =========================================================

print(
    "\n========================================="
)

print(
    "--- FINAL MISSION SUMMARY ---"
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
    "Optimized Mars phase [deg]:",
    mars_phase_deg,
)

print(
    "Mars periapsis distance [km]:",
    result["periapsis_distance_km"],
)

print(
    "Mars periapsis altitude [km]:",
    result["periapsis_altitude_km"],
)

print(
    "Target Mars altitude [km]:",
    target_mars_altitude_km,
)

print(
    "Closest approach [days]:",
    result["closest_approach_days"],
)

print(
    "Relative speed at periapsis [km/s]:",
    result["relative_speed_km_s"],
)

print(
    "Earth-Ship distance after 5 days [km]:",
    result["earth_distance_day5_km"],
)

print(
    "Earth-relative speed after 5 days [km/s]:",
    result["earth_speed_day5_km_s"],
)


print(
    "\n--- Integration ---"
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


print(
    "\n--- Stored trajectory ---"
)

print(
    "Saved trajectory points:",
    len(
        result["records"]
    ),
)

print(
    "Output file:",
    output_file,
)