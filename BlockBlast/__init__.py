from gymnasium.envs.registration import register

__version__ = "0.0.1"

register(
    id="BlockBlast/BlockBlast-v0",
    entry_point="BlockBlast.envs:BlockBLastEnv",
)
