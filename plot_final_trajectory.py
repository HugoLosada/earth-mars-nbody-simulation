import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 1. PATHS
# =========================================================

results_directory = Path(__file__).resolve().parent / "results"

trajectory_file = (
    results_directory
    / "final_trajectory.csv"
)

plots_directory = (
    results_directory
    / "plots"
)

plots_directory.mkdir(
    exist_ok=True
)


# =========================================================
# 2. PHYSICAL CONSTANTS USED FOR PLOTTING
# =========================================================

earth_radius_km = 6371.0
mars_radius_km = 3389.5

earth_parking_altitude_km = 300.0

earth_parking_radius_km = (
    earth_radius_km
    + earth_parking_altitude_km
)

mars_target_altitude_km = 300.0

mars_target_radius_km = (
    mars_radius_km
    + mars_target_altitude_km
)


# =========================================================
# 3. READ TRAJECTORY CSV
# =========================================================

def load_trajectory(filename):
    """
    Load the sampled final trajectory from CSV.

    Numerical columns are converted to NumPy arrays.
    Text columns remain as strings.
    """

    rows = []

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        for row in reader:

            rows.append(
                row
            )


    if len(rows) == 0:

        raise ValueError(
            "Trajectory CSV is empty."
        )


    numeric_columns = [
        "time_days",
        "is_periapsis",

        "sun_x_au",
        "sun_y_au",

        "earth_x_au",
        "earth_y_au",

        "mars_x_au",
        "mars_y_au",

        "ship_x_au",
        "ship_y_au",

        "ship_earth_x_km",
        "ship_earth_y_km",

        "earth_ship_distance_km",
        "earth_relative_speed_km_s",

        "ship_mars_x_km",
        "ship_mars_y_km",

        "mars_ship_distance_km",
        "mars_relative_speed_km_s",
    ]


    data = {}


    for column in numeric_columns:

        data[column] = np.array(
            [
                float(row[column])
                for row in rows
            ]
        )


    data["phase"] = np.array(
        [
            row["phase"]
            for row in rows
        ]
    )


    return data


# =========================================================
# 4. LOAD DATA
# =========================================================

data = load_trajectory(
    trajectory_file
)


number_of_points = len(
    data["time_days"]
)


print(
    "Trajectory points loaded:",
    number_of_points,
)


# =========================================================
# 5. FIND EXACT PERIAPSIS RECORD
# =========================================================

periapsis_mask = (
    data["is_periapsis"]
    == 1
)


if not np.any(
    periapsis_mask
):

    raise ValueError(
        "No periapsis record was found "
        "in final_trajectory.csv."
    )


periapsis_index = np.where(
    periapsis_mask
)[0][0]


periapsis_time_days = (
    data["time_days"][
        periapsis_index
    ]
)

periapsis_distance_km = (
    data["mars_ship_distance_km"][
        periapsis_index
    ]
)

periapsis_altitude_km = (
    periapsis_distance_km
    - mars_radius_km
)

periapsis_relative_speed_km_s = (
    data["mars_relative_speed_km_s"][
        periapsis_index
    ]
)


print(
    "Periapsis time [days]:",
    periapsis_time_days,
)

print(
    "Periapsis altitude [km]:",
    periapsis_altitude_km,
)

print(
    "Periapsis relative speed [km/s]:",
    periapsis_relative_speed_km_s,
)


# =========================================================
# 6. FIGURE 1 — COMPLETE N-BODY TRANSFER
# =========================================================

def plot_complete_transfer(data):
    """
    Plot Earth, Mars and spacecraft trajectories
    over the complete simulated mission.
    """

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )


    # -----------------------------------------------------
    # Trajectories
    # -----------------------------------------------------

    earth_line, = ax.plot(
        data["earth_x_au"],
        data["earth_y_au"],
        linewidth=1.5,
        label="Earth trajectory",
    )


    mars_line, = ax.plot(
        data["mars_x_au"],
        data["mars_y_au"],
        linewidth=1.5,
        label="Mars trajectory",
    )


    ship_line, = ax.plot(
        data["ship_x_au"],
        data["ship_y_au"],
        linewidth=2.2,
        label="Spacecraft trajectory",
    )


    # -----------------------------------------------------
    # Sun
    # -----------------------------------------------------

    ax.scatter(
        data["sun_x_au"][0],
        data["sun_y_au"][0],
        s=130,
        marker="*",
        label="Sun",
        zorder=6,
    )


    # -----------------------------------------------------
    # Initial Earth position
    # -----------------------------------------------------

    ax.scatter(
        data["earth_x_au"][0],
        data["earth_y_au"][0],
        s=65,
        marker="o",
        color=earth_line.get_color(),
        zorder=7,
    )


    # -----------------------------------------------------
    # Initial Mars position
    # -----------------------------------------------------

    ax.scatter(
        data["mars_x_au"][0],
        data["mars_y_au"][0],
        s=65,
        marker="o",
        color=mars_line.get_color(),
        zorder=7,
    )


    # -----------------------------------------------------
    # Spacecraft initial point
    # -----------------------------------------------------

    ax.scatter(
        data["ship_x_au"][0],
        data["ship_y_au"][0],
        s=45,
        marker="o",
        color=ship_line.get_color(),
        zorder=8,
    )


    # -----------------------------------------------------
    # Mars periapsis
    # -----------------------------------------------------

    ax.scatter(
        data["ship_x_au"][
            periapsis_index
        ],
        data["ship_y_au"][
            periapsis_index
        ],
        s=100,
        marker="x",
        label="Mars periapsis",
        zorder=9,
    )


    # -----------------------------------------------------
    # Labels for start and encounter
    # -----------------------------------------------------

    ax.annotate(
        "Departure",
        (
            data["ship_x_au"][0],
            data["ship_y_au"][0],
        ),
        xytext=(10, -25),
        textcoords="offset points",
    )


    ax.annotate(
        "Mars flyby",
        (
            data["ship_x_au"][
                periapsis_index
            ],
            data["ship_y_au"][
                periapsis_index
            ],
        ),
        xytext=(15, 15),
        textcoords="offset points",
    )


    # -----------------------------------------------------
    # Formatting
    # -----------------------------------------------------

    ax.set_title(
        "Earth–Mars Transfer — N-body Simulation"
    )

    ax.set_xlabel(
        "x [AU]"
    )

    ax.set_ylabel(
        "y [AU]"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()


    output_file = (
        plots_directory
        / "heliocentric_transfer.png"
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        output_file,
    )


# =========================================================
# 7. FIGURE 2 — EARTH ESCAPE CLOSE-UP
# =========================================================

def plot_earth_escape(data):
    """
    Plot the early hyperbolic escape relative to Earth.
    """

    maximum_plot_distance_km = (
        60_000.0
    )


    mask = (
        data[
            "earth_ship_distance_km"
        ]
        <= maximum_plot_distance_km
    )


    x = (
        data["ship_earth_x_km"][
            mask
        ]
    )

    y = (
        data["ship_earth_y_km"][
            mask
        ]
    )


    fig, ax = plt.subplots(
        figsize=(8, 8)
    )


    # -----------------------------------------------------
    # Spacecraft trajectory
    # -----------------------------------------------------

    trajectory_line, = ax.plot(
        x,
        y,
        linewidth=2.2,
        label="Spacecraft trajectory",
    )


    # -----------------------------------------------------
    # Earth surface
    # -----------------------------------------------------

    earth_circle = plt.Circle(
        (0.0, 0.0),
        earth_radius_km,
        fill=False,
        linewidth=2.0,
        label="Earth surface",
    )


    ax.add_patch(
        earth_circle
    )


    # -----------------------------------------------------
    # Parking orbit
    # -----------------------------------------------------

    parking_orbit = plt.Circle(
        (0.0, 0.0),
        earth_parking_radius_km,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        label="300 km parking orbit",
    )


    ax.add_patch(
        parking_orbit
    )


    # -----------------------------------------------------
    # TMI point
    # -----------------------------------------------------

    initial_x = (
        data["ship_earth_x_km"][0]
    )

    initial_y = (
        data["ship_earth_y_km"][0]
    )


    ax.scatter(
        initial_x,
        initial_y,
        s=70,
        marker="o",
        color=trajectory_line.get_color(),
        label="TMI point",
        zorder=6,
    )


    ax.annotate(
        "Trans-Mars Injection",
        (
            initial_x,
            initial_y,
        ),
        xytext=(15, -25),
        textcoords="offset points",
    )


    # -----------------------------------------------------
    # Formatting
    # -----------------------------------------------------

    ax.set_title(
        "Hyperbolic Escape from Earth"
    )

    ax.set_xlabel(
        "Earth-relative x [km]"
    )

    ax.set_ylabel(
        "Earth-relative y [km]"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )

    ax.margins(
        0.08
    )

    fig.tight_layout()


    output_file = (
        plots_directory
        / "earth_escape_closeup.png"
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        output_file,
    )


# =========================================================
# 8. FIGURE 3 — MARS FLYBY CLOSE-UP
# =========================================================

def plot_mars_flyby(data):
    """
    Plot a close-up of the optimized flyby relative to Mars.
    """

    maximum_plot_distance_km = (
        20_000.0
    )


    mask = (
        data[
            "mars_ship_distance_km"
        ]
        <= maximum_plot_distance_km
    )


    x = (
        data["ship_mars_x_km"][
            mask
        ]
    )

    y = (
        data["ship_mars_y_km"][
            mask
        ]
    )


    fig, ax = plt.subplots(
        figsize=(8, 8)
    )


    # -----------------------------------------------------
    # Spacecraft trajectory
    # -----------------------------------------------------

    trajectory_line, = ax.plot(
        x,
        y,
        linewidth=2.2,
        label="Spacecraft trajectory",
    )


    # -----------------------------------------------------
    # Mars surface
    # -----------------------------------------------------

    mars_circle = plt.Circle(
        (0.0, 0.0),
        mars_radius_km,
        fill=False,
        linewidth=2.0,
        label="Mars surface",
    )


    ax.add_patch(
        mars_circle
    )


    # -----------------------------------------------------
    # 300 km target altitude
    # -----------------------------------------------------

    target_circle = plt.Circle(
        (0.0, 0.0),
        mars_target_radius_km,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        label="300 km target altitude",
    )


    ax.add_patch(
        target_circle
    )


    # -----------------------------------------------------
    # Exact periapsis
    # -----------------------------------------------------

    periapsis_x = (
        data["ship_mars_x_km"][
            periapsis_index
        ]
    )

    periapsis_y = (
        data["ship_mars_y_km"][
            periapsis_index
        ]
    )


    ax.scatter(
        periapsis_x,
        periapsis_y,
        s=110,
        marker="x",
        color=trajectory_line.get_color(),
        label="Periapsis",
        zorder=7,
    )


    ax.annotate(
        f"Periapsis\n"
        f"{periapsis_altitude_km:.1f} km altitude",
        (
            periapsis_x,
            periapsis_y,
        ),
        xytext=(20, 20),
        textcoords="offset points",
    )


    # -----------------------------------------------------
    # Formatting
    # -----------------------------------------------------

    ax.set_title(
        "Optimized Mars Flyby"
    )

    ax.set_xlabel(
        "Mars-relative x [km]"
    )

    ax.set_ylabel(
        "Mars-relative y [km]"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )

    ax.margins(
        0.08
    )

    fig.tight_layout()


    output_file = (
        plots_directory
        / "mars_flyby_closeup.png"
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        output_file,
    )


# =========================================================
# 9. FIGURE 4 — GLOBAL DISTANCE TO MARS
# =========================================================

def plot_mars_distance(data):
    """
    Plot spacecraft-Mars distance over the full mission.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    distance_line, = ax.plot(
        data["time_days"],
        data["mars_ship_distance_km"],
        linewidth=1.8,
        label="Spacecraft–Mars distance",
    )


    # -----------------------------------------------------
    # Mars surface
    # -----------------------------------------------------

    ax.axhline(
        mars_radius_km,
        linestyle="--",
        linewidth=1.4,
        label="Mars surface",
    )


    # -----------------------------------------------------
    # Target altitude
    # -----------------------------------------------------

    ax.axhline(
        mars_target_radius_km,
        linestyle=":",
        linewidth=1.5,
        label="300 km target altitude",
    )


    # -----------------------------------------------------
    # Periapsis
    # -----------------------------------------------------

    ax.scatter(
        periapsis_time_days,
        periapsis_distance_km,
        s=90,
        marker="x",
        color=distance_line.get_color(),
        label="Periapsis",
        zorder=6,
    )


    # -----------------------------------------------------
    # Formatting
    # -----------------------------------------------------

    ax.set_title(
        "Spacecraft–Mars Distance During Mission"
    )

    ax.set_xlabel(
        "Mission time [days]"
    )

    ax.set_ylabel(
        "Distance from Mars centre [km]"
    )

    ax.set_yscale(
        "log"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()


    output_file = (
        plots_directory
        / "mars_distance_vs_time.png"
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        output_file,
    )


# =========================================================
# 10. FIGURE 5 — MARS ENCOUNTER DETAIL
# =========================================================

def plot_mars_encounter_detail(data):
    """
    Plot altitude above Mars surface during the final
    hours around periapsis.
    """

    time_from_periapsis_hours = (
        data["time_days"]
        - periapsis_time_days
    ) * 24.0


    altitude_km = (
        data["mars_ship_distance_km"]
        - mars_radius_km
    )


    # Show six hours before and after periapsis.
    mask = (
        np.abs(
            time_from_periapsis_hours
        )
        <= 6.0
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    encounter_line, = ax.plot(
        time_from_periapsis_hours[
            mask
        ],
        altitude_km[
            mask
        ],
        linewidth=2.0,
        label="Spacecraft altitude",
    )


    # -----------------------------------------------------
    # Mars surface
    # -----------------------------------------------------

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.4,
        label="Mars surface",
    )


    # -----------------------------------------------------
    # 300 km target
    # -----------------------------------------------------

    ax.axhline(
        mars_target_altitude_km,
        linestyle=":",
        linewidth=1.5,
        label="300 km target altitude",
    )


    # -----------------------------------------------------
    # Periapsis
    # -----------------------------------------------------

    ax.scatter(
        0.0,
        periapsis_altitude_km,
        s=100,
        marker="x",
        color=encounter_line.get_color(),
        label="Periapsis",
        zorder=6,
    )


    ax.annotate(
        f"{periapsis_altitude_km:.2f} km",
        (
            0.0,
            periapsis_altitude_km,
        ),
        xytext=(15, 15),
        textcoords="offset points",
    )


    # -----------------------------------------------------
    # Formatting
    # -----------------------------------------------------

    ax.set_title(
        "Mars Encounter — Periapsis Detail"
    )

    ax.set_xlabel(
        "Time from periapsis [hours]"
    )

    ax.set_ylabel(
        "Altitude above Mars surface [km]"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()


    output_file = (
        plots_directory
        / "mars_encounter_detail.png"
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        output_file,
    )


# =========================================================
# 11. GENERATE ALL FINAL FIGURES
# =========================================================

print(
    "\n--- Generating final figures ---"
)


plot_complete_transfer(
    data
)

plot_earth_escape(
    data
)

plot_mars_flyby(
    data
)

plot_mars_distance(
    data
)

plot_mars_encounter_detail(
    data
)


print(
    "\nAll final figures generated."
)

print(
    "Output directory:",
    plots_directory,
)