def velocity_verlet_step(bodies, dt, acceleration_function):
    """
    Advance all bodies by one time step using the Velocity Verlet method.

    Parameters
    ----------
    bodies : list[Body]
        Bodies participating in the simulation.

    dt : float
        Time step in years.

    acceleration_function : callable
        Function that computes the gravitational acceleration
        acting on a target body.

    Returns
    -------
    None
        The bodies are updated in place.
    """

    old_accelerations = {}

    # 1. Compute accelerations at the current positions.
    for body in bodies:
        old_accelerations[body] = acceleration_function(
            body,
            bodies,
        )

    # 2. Update positions.
    for body in bodies:
        body.position += (
            body.velocity * dt
            + 0.5 * old_accelerations[body] * dt**2
        )

    # 3. Recompute accelerations at the new positions.
    new_accelerations = {}

    for body in bodies:
        new_accelerations[body] = acceleration_function(
            body,
            bodies,
        )

    # 4. Update velocities.
    for body in bodies:
        body.velocity += (
            0.5
            * (
                old_accelerations[body]
                + new_accelerations[body]
            )
            * dt
        )