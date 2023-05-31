


from .autobots_model import AutoBotsModel

__all__ = {
    'AutoBotsModel': AutoBotsModel,
}


def build_autobots_model(config):
    model = __all__[config.NAME](
        config=config
    )

    return model