import numpy as np






class Body:
    """
    Represents a physical body in the orbital simulation.

    Parameters
    ----------
    name : str
        Name of the body.
    mass : float
        Mass in solar masses.
    position : tuple or list
        Initial 2D position in astronomical units (AU).
    velocity : tuple or list
        Initial 2D velocity in AU/year.
    """
    def __init__(self, name, mass, position, velocity):
        self.name = name 
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        


