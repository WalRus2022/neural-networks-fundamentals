import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        ### YOUR CODE HERE
        return -ctx.strength * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            gradient_reversal_strength=1.0,
            activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
        ):
        ## YOUR CODE HERE
        torch.nn.Module.__init__(self)

        noise_dim, hidden_dim, image_dim = channels

        self.generator_discriminator_bridge = GradientReversalLayer(
            gradient_reversal_strength
        )
        self.gradient_reversal = self.generator_discriminator_bridge

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(noise_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, image_dim),
            torch.nn.Tanh(),
        )

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(image_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, noise_dim),
        )

        self.classifier = torch.nn.Linear(noise_dim, 1)

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        ## YOUR CODE HERE
        noise = batch['data']['noise']
        real = batch['data'].get('real', batch['data'].get('image'))

        generated = self.generator(noise)
        reversed_generated = self.generator_discriminator_bridge(generated)

        real = real.reshape(real.shape[0], -1)
        discriminator_input = torch.cat([reversed_generated, real], dim=0)

        discriminator_logits = self.discriminate(discriminator_input)

        fake_logits = discriminator_logits[:noise.shape[0]]
        real_logits = discriminator_logits[noise.shape[0]:]

        if 'signals' not in batch:
            batch['signals'] = {}
        if 'postprocessed' not in batch:
            batch['postprocessed'] = {}

        batch['signals']['generated'] = generated.reshape(noise.shape[0], 28, 28)
        batch['signals']['discriminator_logits'] = discriminator_logits
        batch['signals']['fake_logits'] = fake_logits
        batch['signals']['real_logits'] = real_logits
        batch['signals']['discriminator_scores'] = discriminator_logits
        batch['signals']['fake_scores'] = fake_logits
        batch['signals']['real_scores'] = real_logits

        batch['postprocessed']['discriminator_logits'] = discriminator_logits
        batch['postprocessed']['discriminator_scores'] = discriminator_logits
        batch['postprocessed']['discriminator_probabilities'] = torch.sigmoid(discriminator_logits)

        batch['postprocessed']['fake_logits'] = fake_logits
        batch['postprocessed']['fake_scores'] = fake_logits
        batch['postprocessed']['fake_probabilities'] = torch.sigmoid(fake_logits)

        batch['postprocessed']['real_logits'] = real_logits
        batch['postprocessed']['real_scores'] = real_logits
        batch['postprocessed']['real_probabilities'] = torch.sigmoid(real_logits)

        return batch