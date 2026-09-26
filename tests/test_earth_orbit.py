import numpy as np

from bodies import Body
from physics import (
    circular_orbit_speed,
    gravitational_acceleration,
    total_energy
)
from integrators import velocity_verlet_step


# Orbital radius of Earth in astronomical units (AU)
earth_radius = 1.0

# Circular orbital speed of Earth around the Sun
earth_speed = circular_orbit_speed(
    central_mass=1.0,
    radius=earth_radius,
)


# Create the Sun
sun = Body(
    name="Sun",
    mass=1.0,
    position=(0.0, 0.0),
    velocity=(0.0, 0.0),
)


# Create the Earth
earth = Body(
    name="Earth",
    mass=3.004e-6,
    position=(earth_radius, 0.0),
    velocity=(0.0, earth_speed),
)


# Bodies included in the simulation
bodies = [sun, earth]


# Simulation parameters
simulation_time = 1.0
dt = 0.001

# Number of integration steps
number_of_steps = int(
    simulation_time / dt
)


# Save the initial Earth position relative to the Sun
initial_relative_position = (
    earth.position - sun.position
).copy()

initial_energy = total_energy(bodies)

energy_errors = []

for step in range(number_of_steps):
    velocity_verlet_step(
        bodies,
        dt,
        gravitational_acceleration,
    )

    current_energy = total_energy(bodies)

    relative_error = abs(
        (current_energy - initial_energy)
        / initial_energy
    )

    energy_errors.append(relative_error)

max_energy_error = max(energy_errors)

# Calculate the final Earth position relative to the Sun
final_relative_position = (
    earth.position - sun.position
)


# Calculate the final Earth-Sun distance
final_distance = np.linalg.norm(
    final_relative_position
)


# Difference between the final orbital radius and 1 AU
radial_error = abs(
    final_distance - earth_radius
)


# Difference between the initial and final relative positions
position_error = np.linalg.norm(
    final_relative_position
    - initial_relative_position
)


# Display results
print(
    "Initial relative position:",
    initial_relative_position,
)

print(
    "Final relative position:",
    final_relative_position,
)

print(
    "Final Earth-Sun distance:",
    final_distance,
)

print(
    "Radial error:",
    radial_error,
)

print(
    "Position error after one year:",
    position_error,
)

initial_energy = total_energy(bodies)
final_energy = total_energy(bodies)

relative_energy_error = abs(
    (final_energy - initial_energy)
    / initial_energy
)

print("Initial energy:", initial_energy)
print("Final energy:", final_energy)
print(
    "Relative energy error:",
    relative_energy_error,
)

print(
    "Maximum relative energy error:",
    max_energy_error,
)