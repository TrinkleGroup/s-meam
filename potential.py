class Potential(object):
    """Potential class for classical MD potentials"""

    def __init__(self):
        self.cutoff = 0.0

    @property
    def cutoff(self):
        """Potential energy cutoff radius"""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, cutoff):
        if cutoff < 0:
            raise ValueError, "Cannot set cutoff to be negative."
        else:
            self._cutoff = cutoff

    def compute_energies(self):
        raise NotImplementedError

    def compute_forces(self):
        raise NotImplementedError
