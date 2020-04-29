
class GAN():
    """
    Abstract GAN Class definition
    
    """
    def __init__(self):
        self.discriminator = None
        self.generator = None

    def _build_generator(self):
        pass
    
    def _build_discriminator(self):
        pass

    def train_step(self):
        pass



