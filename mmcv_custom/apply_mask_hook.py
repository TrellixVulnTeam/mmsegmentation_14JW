import mmcv
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ApplyMaskHook(Hook):
    """
    Customized masking operation.
    """

    def before_train_iter(self, runner):
        """
        Apply mask before each update
        """
        runner.model.backbone.apply_mssk()
        print('apply masks before training')

    def after_train_iter(self, runner):
        """
        Apply mask after each update
        """
        runner.model.backbone.apply_mssk()
        print('apply masks after training')