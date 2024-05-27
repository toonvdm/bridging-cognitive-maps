class Memory:
    def __init__(self):
        self.episodes = []

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        return episode["observations"], episode["actions"]

    def add_episode(self, observations, actions):
        self.episodes.append(
            {"observations": observations, "actions": actions}
        )
