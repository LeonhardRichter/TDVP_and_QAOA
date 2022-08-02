class tdvp_result():
    def __init__(self) -> None:
        pass

    @property
    def num_calls(self):
        """The num_calls property."""
        return self._num_calls

    @num_calls.setter
    def num_calls(self, value):
        self._num_calls = value

    @property
    def opt_pars(self):
        """The opt_pars property."""
        return self._opt_pars

    @opt_pars.setter
    def opt_pars(self, value):
        self._opt_pars = value

    @property
    def opt_state(self):
        """The opt_state property."""
        return self._opt_state

    @opt_state.setter
    def opt_state(self, value):
        self._opt_state = value

    @property
    def num_steps(self):
        """The num_steps property."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value):
        self._num_steps = value

    @property
    def opt_goal_val(self):
        """The opt_goal_val property."""
        return self._opt_goal_val

    @opt_goal_val.setter
    def opt_goal_val(self, value):
        self._opt_goal_val = value


class tdvp_optimizer():
    def __init__(self, psi, H) -> None:
        pass

    def flow(pars, stepsize) -> tuple:
        """flow according to tdvp from pars for stepsize duration.

        Args:
            pars (tuple): starting point parameters
            stepsize (float): stepsize of the flow

        Returns:
            _type_: _description_
        """
        pass

    def optimize(tol, stepsize) -> tdvp_result:
        """Run the optimization algorithm using the tdvp flow.

        Args:
            tol (float): _description_
            stepsize (float): _description_

        Returns:
            tdvp_result: _description_
        """
        pass

    def gram_eval(pars):
        """evaluate the gram matrix on a quantum circuit.

        Args:
            pars (_type_): _description_

        Returns:
            _type_: _description_
        """
