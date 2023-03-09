# Reference: https://github.com/LMescheder/GAN_stability/blob/c1f64c9efeac371453065e5ce71860f4c2b97357/gan_training/train.py#L118
def toggle_grad(model, requires_grad, toggle_embedding=True):
    for name, p in model.named_parameters():
        if ('embedding' not in name) or toggle_embedding:
            p.requires_grad_(requires_grad)


# Reference: https://www.programmersought.com/article/28492072406/
class ExponentialMovingAverage:
    def __init__(self, model, decay, skeleton=False):
        self.model = model
        self.decay = decay
        self.skeleton = skeleton

        self.shadow = {}
        self.backup = {}

    def register(self, model=None):
        model = self.model if model is None else self.model  # for loading and resuming training
        if not self.skeleton:
            for name, param in model.named_parameters():
                self.shadow[name] = param.data.clone()
        return self

    def update(self):
        if not self.skeleton:
            toggle_grad(self.model, False)  # FIXME: Is it necessary? This is error-prone due to toggle_embedding
            for name, param in self.model.named_parameters():
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        return self

    def apply_shadow(self):
        if not self.skeleton:
            for name, param in self.model.named_parameters():
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
        return self

    def restore(self):
        if not self.skeleton:
            for name, param in self.model.named_parameters():
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}
        return self
