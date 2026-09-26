import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


# =========================================================
# 1. FILE PATHS
# =========================================================

results_directory = Path(__file__).resolve().parent / "results"

trajectory_file = (
    results_directory
    / "final_trajectory.csv"
)

animation_file = (
    results_directory
    / "earth_mars_transfer.gif"
)


# =========================================================
# 2. ANIMATION SETTINGS
# =========================================================

# Number of frames used for the animation.
#
# Increasing this makes the animation smoother,
# but also slower to generate and larger if saved.
NUMBER_OF_FRAMES = 700


# Delay between frames in milliseconds.
#
# 30 ms corresponds to roughly 33 frames per second.
FRAME_INTERVAL_MS = 30


# Set this to True if you also want to save
# the animation as a GIF.
SAVE_GIF = False


# =========================================================
# 3. LOAD TRAJECTORY FROM CSV
# =========================================================

def load_trajectory(filename):
    """
    Load the final simulated trajectory.

    Only the quantities needed for the heliocentric
    animation are read.
    """

    time_days = []

    earth_x = []
    earth_y = []

    mars_x = []
    mars_y = []

    ship_x = []
    ship_y = []


    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        for row in reader:

            time_days.append(
                float(
                    row["time_days"]
                )
            )

            earth_x.append(
                float(
                    row["earth_x_au"]
                )
            )

            earth_y.append(
                float(
                    row["earth_y_au"]
                )
            )

            mars_x.append(
                float(
                    row["mars_x_au"]
                )
            )

            mars_y.append(
                float(
                    row["mars_y_au"]
                )
            )

            ship_x.append(
                float(
                    row["ship_x_au"]
                )
            )

            ship_y.append(
                float(
                    row["ship_y_au"]
                )
            )


    return {
        "time_days":
            np.array(time_days),

        "earth_x":
            np.array(earth_x),

        "earth_y":
            np.array(earth_y),

        "mars_x":
            np.array(mars_x),

        "mars_y":
            np.array(mars_y),

        "ship_x":
            np.array(ship_x),

        "ship_y":
            np.array(ship_y),
    }


# =========================================================
# 4. LOAD THE FINAL SIMULATION
# =========================================================

data = load_trajectory(
    trajectory_file
)


print(
    "Trajectory points loaded:",
    len(
        data["time_days"]
    ),
)


# =========================================================
# 5. REMOVE POSSIBLE DUPLICATE TIMES
# =========================================================

# The trajectory was saved using different sampling
# frequencies during Earth escape, cruise and Mars flyby.
#
# For the animation we first make sure each time appears
# only once.

unique_times, unique_indices = np.unique(
    data["time_days"],
    return_index=True,
)


time_days = (
    data["time_days"][
        unique_indices
    ]
)

earth_x = (
    data["earth_x"][
        unique_indices
    ]
)

earth_y = (
    data["earth_y"][
        unique_indices
    ]
)

mars_x = (
    data["mars_x"][
        unique_indices
    ]
)

mars_y = (
    data["mars_y"][
        unique_indices
    ]
)

ship_x = (
    data["ship_x"][
        unique_indices
    ]
)

ship_y = (
    data["ship_y"][
        unique_indices
    ]
)


# =========================================================
# 6. CREATE UNIFORM ANIMATION TIMES
# =========================================================

# The original CSV does NOT have uniform time spacing:
#
# Earth escape  -> samples every few minutes
# Cruise        -> samples every several hours
# Mars flyby    -> samples every few seconds
#
# If we animated the CSV rows directly, the apparent
# speed of time would change dramatically.
#
# Therefore we create equally spaced mission times
# and interpolate the positions.

animation_times = np.linspace(
    time_days[0],
    time_days[-1],
    NUMBER_OF_FRAMES,
)


# =========================================================
# 7. INTERPOLATE ALL BODY POSITIONS
# =========================================================

earth_x_animation = np.interp(
    animation_times,
    time_days,
    earth_x,
)

earth_y_animation = np.interp(
    animation_times,
    time_days,
    earth_y,
)


mars_x_animation = np.interp(
    animation_times,
    time_days,
    mars_x,
)

mars_y_animation = np.interp(
    animation_times,
    time_days,
    mars_y,
)


ship_x_animation = np.interp(
    animation_times,
    time_days,
    ship_x,
)

ship_y_animation = np.interp(
    animation_times,
    time_days,
    ship_y,
)


# =========================================================
# 8. CREATE FIGURE
# =========================================================

fig, ax = plt.subplots(
    figsize=(9, 9)
)


# =========================================================
# 9. DETERMINE AUTOMATIC AXIS LIMITS
# =========================================================

all_x = np.concatenate(
    [
        earth_x_animation,
        mars_x_animation,
        ship_x_animation,
    ]
)

all_y = np.concatenate(
    [
        earth_y_animation,
        mars_y_animation,
        ship_y_animation,
    ]
)


x_min = np.min(
    all_x
)

x_max = np.max(
    all_x
)

y_min = np.min(
    all_y
)

y_max = np.max(
    all_y
)


x_range = (
    x_max
    - x_min
)

y_range = (
    y_max
    - y_min
)


margin = 0.10


ax.set_xlim(
    x_min
    - margin * x_range,

    x_max
    + margin * x_range,
)

ax.set_ylim(
    y_min
    - margin * y_range,

    y_max
    + margin * y_range,
)


# =========================================================
# 10. STATIC REFERENCE TRAJECTORIES
# =========================================================

# These faint curves show the path each planet follows
# during the simulated mission.

earth_reference, = ax.plot(
    earth_x_animation,
    earth_y_animation,
    linewidth=1.0,
    alpha=0.35,
    label="Earth trajectory",
)


mars_reference, = ax.plot(
    mars_x_animation,
    mars_y_animation,
    linewidth=1.0,
    alpha=0.35,
    label="Mars trajectory",
)


# =========================================================
# 11. SPACECRAFT TRAIL
# =========================================================

ship_trail, = ax.plot(
    [],
    [],
    linewidth=2.0,
    label="Spacecraft trajectory",
)


# =========================================================
# 12. CURRENT BODY POSITIONS
# =========================================================

earth_marker, = ax.plot(
    [],
    [],
    marker="o",
    linestyle="",
    markersize=8,
    color=earth_reference.get_color(),
)


mars_marker, = ax.plot(
    [],
    [],
    marker="o",
    linestyle="",
    markersize=8,
    color=mars_reference.get_color(),
)


ship_marker, = ax.plot(
    [],
    [],
    marker="o",
    linestyle="",
    markersize=6,
    color=ship_trail.get_color(),
)


# =========================================================
# 13. SUN
# =========================================================

sun_marker, = ax.plot(
    [0.0],
    [0.0],
    marker="*",
    linestyle="",
    markersize=14,
    label="Sun",
)


# =========================================================
# 14. MISSION TIME TEXT
# =========================================================

mission_time_text = ax.text(
    0.03,
    0.97,
    "",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=11,
)


# =========================================================
# 15. SPACECRAFT-MARS DISTANCE TEXT
# =========================================================

mars_distance_text = ax.text(
    0.03,
    0.92,
    "",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=10,
)


# =========================================================
# 16. GRAPH FORMATTING
# =========================================================

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
    loc="lower left"
)

fig.tight_layout()


# =========================================================
# 17. INITIALIZATION FUNCTION
# =========================================================

def initialize_animation():
    """
    Clear all animated objects before frame 0.
    """

    ship_trail.set_data(
        [],
        [],
    )

    earth_marker.set_data(
        [],
        [],
    )

    mars_marker.set_data(
        [],
        [],
    )

    ship_marker.set_data(
        [],
        [],
    )

    mission_time_text.set_text(
        ""
    )

    mars_distance_text.set_text(
        ""
    )


    return (
        ship_trail,
        earth_marker,
        mars_marker,
        ship_marker,
        mission_time_text,
        mars_distance_text,
    )


# =========================================================
# 18. UPDATE FUNCTION
# =========================================================

def update_animation(frame):
    """
    Update the position of each object for one animation
    frame.
    """

    # -----------------------------------------------------
    # Current coordinates
    # -----------------------------------------------------

    current_earth_x = (
        earth_x_animation[
            frame
        ]
    )

    current_earth_y = (
        earth_y_animation[
            frame
        ]
    )


    current_mars_x = (
        mars_x_animation[
            frame
        ]
    )

    current_mars_y = (
        mars_y_animation[
            frame
        ]
    )


    current_ship_x = (
        ship_x_animation[
            frame
        ]
    )

    current_ship_y = (
        ship_y_animation[
            frame
        ]
    )


    # -----------------------------------------------------
    # Current body markers
    # -----------------------------------------------------

    earth_marker.set_data(
        [current_earth_x],
        [current_earth_y],
    )


    mars_marker.set_data(
        [current_mars_x],
        [current_mars_y],
    )


    ship_marker.set_data(
        [current_ship_x],
        [current_ship_y],
    )


    # -----------------------------------------------------
    # Spacecraft trail
    # -----------------------------------------------------

    ship_trail.set_data(
        ship_x_animation[
            :frame + 1
        ],

        ship_y_animation[
            :frame + 1
        ],
    )


    # -----------------------------------------------------
    # Mission time
    # -----------------------------------------------------

    current_time_days = (
        animation_times[
            frame
        ]
    )


    mission_time_text.set_text(
        f"Mission time: "
        f"{current_time_days:.1f} days"
    )


    # -----------------------------------------------------
    # Spacecraft-Mars distance
    # -----------------------------------------------------

    dx = (
        current_ship_x
        - current_mars_x
    )

    dy = (
        current_ship_y
        - current_mars_y
    )


    distance_au = np.sqrt(
        dx**2
        + dy**2
    )


    AU_TO_KM = (
        149_597_870.7
    )


    distance_km = (
        distance_au
        * AU_TO_KM
    )


    # Display millions of km while far away,
    # but ordinary km close to Mars.

    if (
        distance_km
        >= 1_000_000
    ):

        mars_distance_text.set_text(
            f"Distance to Mars: "
            f"{distance_km / 1_000_000:.1f} million km"
        )

    else:

        mars_distance_text.set_text(
            f"Distance to Mars: "
            f"{distance_km:,.0f} km"
        )


    return (
        ship_trail,
        earth_marker,
        mars_marker,
        ship_marker,
        mission_time_text,
        mars_distance_text,
    )


# =========================================================
# 19. CREATE ANIMATION
# =========================================================

animation = FuncAnimation(
    fig,
    update_animation,
    frames=NUMBER_OF_FRAMES,
    init_func=initialize_animation,
    interval=FRAME_INTERVAL_MS,
    blit=True,
    repeat=False,
)


# =========================================================
# 20. OPTIONAL GIF EXPORT
# =========================================================

if SAVE_GIF:

    print(
        "\nSaving animation..."
    )

    writer = PillowWriter(
        fps=30
    )

    animation.save(
        animation_file,
        writer=writer,
        dpi=100,
    )

    print(
        "Animation saved to:",
        animation_file,
    )


# =========================================================
# 21. DISPLAY ANIMATION
# =========================================================

print(
    "\nStarting animation."
)

print(
    "Mission duration:",
    f"{time_days[-1]:.2f}",
    "days",
)

print(
    "Animation frames:",
    NUMBER_OF_FRAMES,
)


plt.show()