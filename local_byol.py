import torch
import torch.nn as nn

import flair


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    """Copies the parameters of the online network to the momentum network.
    Args:
        online_net (nn.Module): online network (e.g. online backbone, online projection, etc...).
        momentum_net (nn.Module): momentum network (e.g. momentum backbone,
            momentum projection, etc...).
    """

    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class BYOL(nn.Module):
    def __init__(self, model: flair.nn.Model, hidden_dim_factor: float = 1.0, output_dim_factor: float = 1.0):

        super().__init__()
        self.model = model
        embedding_dim = model.embeddings.embedding_length
        hidden_dim = int(model.embeddings.embedding_length * hidden_dim_factor)
        output_dim = int(model.embeddings.embedding_length * output_dim_factor)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, sentences):
        out = 1
        self.model.embeddings.embed(sentences)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out
