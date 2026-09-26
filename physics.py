import numpy as np

G = 4 * np.pi**2


def gravitational_acceleration(target, bodies):
    """
    Calculate the total gravitational acceleration acting on a body.

    The acceleration is computed by summing the Newtonian gravitational
    contribution from every other body in the system:

        a_i = G * sum[m_j * (r_j - r_i) / |r_j - r_i|^3]

    Parameters
    ----------
    target : Body
        Body whose acceleration is being calculated.

    bodies : list[Body]
        All bodies in the gravitational system.

    Returns
    -------
    numpy.ndarray
        Two-dimensional acceleration vector [ax, ay].
    """
    acceleration = np.array([0.0, 0.0])

    for body in bodies:

         # A body must not exert gravitational force on itself.
        if body is target:
            continue

        # Vector pointing from the target body to the attracting body.
        displacement = body.position - target.position
        # Euclidean distance between both bodies.
        distance = np.linalg.norm(displacement)
        # Add this body's gravitational contribution
        acceleration += G * body.mass * displacement / distance**3

    return acceleration

def circular_orbit_speed(central_mass, radius):
    """
    Calculate the orbital speed for a circular orbit.

    Parameters
    ----------
    central_mass : float
        Mass of the central body in solar masses.
    radius : float
        Orbital radius in AU.

    Returns
    -------
    float
        Circular orbital speed in AU/year.
    """
    return np.sqrt(G * central_mass / radius)


def hohmann_transfer_speed(
    central_mass,
    departure_radius,
    arrival_radius,
):
    """
    Calculate the heliocentric departure speed for a Hohmann transfer.

    The spacecraft starts at the periapsis of the transfer ellipse.

    Parameters
    ----------
    central_mass : float
        Mass of the central body in solar masses.
    departure_radius : float
        Radius of the initial circular orbit in AU.
    arrival_radius : float
        Radius of the destination circular orbit in AU.

    Returns
    -------
    float
        Spacecraft speed at departure in AU/year.
    """

    semi_major_axis = (
        departure_radius + arrival_radius
    ) / 2

    return np.sqrt(
        G
        * central_mass
        * (
            2 / departure_radius
            - 1 / semi_major_axis
        )
    )


def hohmann_phase_angle(
    central_mass,
    departure_radius,
    arrival_radius,
):
    """
    Calculate the required initial phase angle of the destination
    body for an outward Hohmann transfer.

    Returns
    -------
    float
        Initial phase angle in radians.
    """

    semi_major_axis = (
        departure_radius + arrival_radius
    ) / 2

    mu = G * central_mass

    transfer_time = (
        np.pi
        * np.sqrt(semi_major_axis**3 / mu)
    )

    destination_angular_speed = np.sqrt(
        mu / arrival_radius**3
    )

    return (
        np.pi
        - destination_angular_speed * transfer_time
    )

def kinetic_energy(body):
    """
    Calculate the kinetic energy of a body.

    Parameters
    ----------
    body : Body
        Body whose kinetic energy is calculated.

    Returns
    -------
    float
        Kinetic energy in the simulation's unit system.
    """

    speed_squared = np.dot(
        body.velocity,
        body.velocity,
    )

    return 0.5 * body.mass * speed_squared


def gravitational_potential_energy(body1, body2):
    """
    Calculate the gravitational potential energy
    between two bodies.
    """

    displacement = (
        body2.position - body1.position
    )

    distance = np.linalg.norm(displacement)

    if distance == 0:
        raise ValueError(
            f"{body1.name} and {body2.name} "
            "occupy the same position."
        )

    return (
        -G
        * body1.mass
        * body2.mass
        / distance
    )

def total_energy(bodies):
    """
    Calculate the total mechanical energy
    of an N-body gravitational system.
    """

    kinetic = 0.0
    potential = 0.0

    for body in bodies:
        kinetic += kinetic_energy(body)

    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            potential += gravitational_potential_energy(
                bodies[i],
                bodies[j],
            )

    return kinetic + potential

def hohmann_transfer_time(
    central_mass,
    departure_radius,
    arrival_radius,
):
    """
    Calculate the travel time for a Hohmann transfer.

    The spacecraft travels half of the transfer ellipse.

    Parameters
    ----------
    central_mass : float
        Mass of the central body in solar masses.

    departure_radius : float
        Radius of the initial circular orbit in AU.

    arrival_radius : float
        Radius of the destination circular orbit in AU.

    Returns
    -------
    float
        Transfer time in years.
    """

    semi_major_axis = (
        departure_radius + arrival_radius
    ) / 2

    mu = G * central_mass

    return (
        np.pi
        * np.sqrt(semi_major_axis**3 / mu)
    )


def circular_orbit_speed_from_mu(mu, radius):
    """
    Calculate circular orbital speed from the gravitational
    parameter mu = G * M.

    Parameters
    ----------
    mu : float
        Gravitational parameter.

    radius : float
        Orbital radius.

    Returns
    -------
    float
        Circular orbital speed.
    """

    return np.sqrt(mu / radius)


def hyperbolic_periapsis_speed(
    mu,
    periapsis_radius,
    v_infinity,
):
    """
    Calculate the periapsis speed required for a hyperbolic
    escape trajectory.

    Parameters
    ----------
    mu : float
        Gravitational parameter of the central body.

    periapsis_radius : float
        Distance from the body's center at the burn point.

    v_infinity : float
        Hyperbolic excess velocity.

    Returns
    -------
    float
        Required periapsis speed.
    """

    return np.sqrt(
        v_infinity**2
        + 2 * mu / periapsis_radius
    )

def hyperbolic_eccentricity(
    mu,
    periapsis_radius,
    v_infinity,
):
    """
    Calculate the eccentricity of a hyperbolic escape trajectory.

    Parameters
    ----------
    mu : float
        Gravitational parameter of the central body.

    periapsis_radius : float
        Distance from the central body at periapsis.

    v_infinity : float
        Hyperbolic excess velocity.

    Returns
    -------
    float
        Hyperbolic eccentricity.
    """

    return (
        1
        + periapsis_radius
        * v_infinity**2
        / mu
    )


def hyperbolic_asymptote_angle(eccentricity):
    """
    Calculate the true anomaly of the asymptote of a hyperbolic orbit.

    Parameters
    ----------
    eccentricity : float
        Hyperbolic eccentricity, e > 1.

    Returns
    -------
    float
        Asymptotic true anomaly in radians.
    """

    if eccentricity <= 1:
        raise ValueError(
            "Hyperbolic eccentricity must be greater than 1."
        )

    return np.arccos(
        -1 / eccentricity
    )