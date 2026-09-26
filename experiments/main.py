import numpy as np

from bodies import Body
from physics import (
    circular_orbit_speed,
    hohmann_transfer_speed,
    hohmann_phase_angle,
)


# Orbital radii in astronomical units (AU)
earth_radius = 1.0
mars_radius = 1.53


# Circular orbital speeds around the Sun
earth_speed = circular_orbit_speed(
    central_mass=1.0,
    radius=earth_radius,
)

mars_speed = circular_orbit_speed(
    central_mass=1.0,
    radius=mars_radius,
)


# Hohmann transfer parameters
ship_speed = hohmann_transfer_speed(
    central_mass=1.0,
    departure_radius=earth_radius,
    arrival_radius=mars_radius,
)

mars_phase = hohmann_phase_angle(
    central_mass=1.0,
    departure_radius=earth_radius,
    arrival_radius=mars_radius,
)


# Create celestial bodies
sun = Body(
    name="Sun",
    mass=1.0,
    position=(0.0, 0.0),
    velocity=(0.0, 0.0),
)

earth = Body(
    name="Earth",
    mass=3.004e-6,
    position=(earth_radius, 0.0),
    velocity=(0.0, earth_speed),
)

mars = Body(
    name="Mars",
    mass=3.214e-7,
    position=(
        mars_radius * np.cos(mars_phase),
        mars_radius * np.sin(mars_phase),
    ),
    velocity=(
        -mars_speed * np.sin(mars_phase),
        mars_speed * np.cos(mars_phase),
    ),
)


# Check calculated values
print("Earth speed:", earth_speed)
print("Mars speed:", mars_speed)
print("Ship Hohmann speed:", ship_speed)
print("Mars phase angle:", np.degrees(mars_phase))

print("Earth position:", earth.position)
print("Earth velocity:", earth.velocity)

print("Mars position:", mars.position)
print("Mars velocity:", mars.velocity)