import logging

from flair.trainers.plugins.base import TrainerPlugin
from flair.training_utils import add_file_handler

log = logging.getLogger("flair")


class LogFilePlugin(TrainerPlugin):
    """
    Plugin for the training.log file
    """

    @TrainerPlugin.hook
    def after_training_setup(self, **kw):
        self.log_handler = add_file_handler(log, self.trainer.base_path / "training.log")

    @TrainerPlugin.hook("_training_exception", "after_teardown")
    def close_file_handler(self, **kw):
        self.log_handler.close()
        log.removeHandler(self.log_handler)