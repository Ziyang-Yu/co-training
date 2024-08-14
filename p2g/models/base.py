import abc

# this defines the need func for the model
# which is impl by each model
# func 1 : prepare lora
# func 2 : load ckpt
class P2GModel:
    @abc.abstractmethod
    def prepare_peft(self, prepare_fn):
        raise NotImplementedError

    @abc.abstractmethod
    def load_peft(self):
        raise NotImplementedError

    @abc.abstractmethod
    def save_peft(self, save_filter):
        raise NotImplementedError

    @abc.abstractmethod
    def load_ckpt(self, path=None):
        r"""
        if path is none, resume from pre-trained
        else load from path
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, input_args):
        raise NotImplementedError
