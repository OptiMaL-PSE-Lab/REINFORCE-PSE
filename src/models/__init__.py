class AbstractModel(object):
    """Contains what is expected to be implemented from any optimization model."""

    def __init__(self, states_dims, controls_dims, parameters):
        self.states_dims = states_dims
        self.controls_dims = controls_dims
        self.parameters = parameters

    def _check_dims(self, state, controls):
        """Runtime check of control and state dimensions."""
        try:
            iter(state)
            iter(controls)
        except TypeError:
            raise TypeError(
                "Please use containers for controls and states: value --> [value]."
            )
        else:
            assert self.controls_dims == len(
                controls
            ), f"This model expects {self.controls_dims} controls!"
            assert self.states_dims == len(
                state
            ), f"This model expects {self.states_dims} controls!"

    def step(self, state, controls, *args, **kwargs):
        """General environment step."""
        raise NotImplementedError(
            "Logic that links current state actions to next state must be specified!"
        )
