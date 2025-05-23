import lightning as L
from loguru import logger
from ..utils import safe_masked_max
from ..utils import safe_weighted_avg


class ESTForEmbedding(L.LightningModule):
    """A PyTorch Lightning Module for extracting embeddings only model."""

    def __init__(self, pooling_method: str, model):
        """Initializes the Lightning Module.

        Args:
            Pooling method: str
            Model loaded from checkpoint.

        """
        super().__init__()

        self.pooling_method = pooling_method
        self.model = model
        self.uses_dep_graph = True if 'Nested' in model.__class__.__name__ else False
        logger.debug(f'dep graph {self.uses_dep_graph}')


    def predict_step(self, batch, batch_idx):
        """Retrieves the embeddings and returns them."""
        encoded = self.model(batch).last_hidden_state
        event_encoded = encoded[:, :, -1, :] if self.uses_dep_graph else encoded

        # `event_encoded` is of shape [batch X seq X hidden_dim]. For pooling, I want to put the sequence
        # dimension as last, so we'll transpose.
        event_encoded = event_encoded.transpose(1, 2)

        match self.pooling_method:
            case "last":
                return event_encoded[:, :, -1]
            case "max":
                return safe_masked_max(event_encoded, batch["event_mask"])
            case "mean":
                return safe_weighted_avg(event_encoded, batch["event_mask"])[0]
            case "none":
                return event_encoded
            case _:
                raise ValueError(f"{self.pooling_method} is not a supported pooling method.")
