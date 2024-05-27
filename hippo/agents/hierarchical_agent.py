import logging

logger = logging.getLogger(__name__)


class HierarchicalAgent:
    def __init__(self, model_pf, model_hippo, fe_threshold, name=""):
        self.model_pf = model_pf
        self.model_hippo = model_hippo

        self.name = name

        self.fe_threshold = fe_threshold
        self.fe_sequence = []
        self.df_sequence = []

    def reset(self):
        self.model_pf.reset()
        self.model_hippo.reset()
        self.model_hippo.set_state_preference(
            self.model_pf.get_hippo_empirical_prior()
        )
        self.fe_sequence = []
        self.df_sequence = []

    @property
    def df(self):
        if len(self.fe_sequence) >= 2:
            return abs(self.fe_sequence[-2] - self.fe_sequence[-1])
        else:
            return 10

    def act(self, obs, reward, prev=None):
        info = dict({})

        # Act in the lowest level, i.e. get some sensory observation and infer the state
        action, hippo_dict = self.model_hippo.act(obs)
        self.fe_sequence.append(hippo_dict["free_energy"])
        info.update(hippo_dict)

        self.df_sequence.append(self.df)

        # the hippocampus model has reached the internal goal
        # and is sending this observation to the prefrontal model
        if (
            self.fe_sequence[-1] < self.fe_threshold
            or len(self.fe_sequence) < 2
        ):
            hippo_state = self.model_hippo.qs.argmax()
            obs = [hippo_state, reward]
            if prev is not None:
                obs.append(prev)
            _, pfc_dict = self.model_pf.act(*obs)
            self.model_hippo.set_state_preference(pfc_dict["lower_state"])
            info.update(pfc_dict)

        return action, info
