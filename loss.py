from random import choice
from torch import Tensor
from torch.nn import TripletMarginLoss
from torch import nonzero, logical_not, cdist, stack


class TripletLoss:
    """
    A wrapper class for `TripletMarginLoss`.

    Args:
        - hard (bool): indicate whether to select the hardest negative and positive examples for each anchor. Default: `false`
    """

    def __init__(self, hard: bool = False):
        self.__base__ = TripletMarginLoss()
        self.hard = hard

    def __call__(self, features: Tensor, labels: Tensor) -> Tensor:
        """
        Calculate triplet loss.

        Args:
            - features (Dataset): embeddings from feature extractor
            - labels (int): labels
        """
        n = len(features)
        labels_mask = labels.expand(n, n).eq(labels.expand(n, n).t())

        positives, negatives = [], []
        if self.hard:  # Find the hardest positive and negative for each anchor
            dist = cdist(features.float(), features.float())
            for index, mask in enumerate(labels_mask):
                p_indices = nonzero(mask).flatten()
                p = features[p_indices[dist[index][p_indices].argmax()]]
                n_indices = nonzero(logical_not(mask)).flatten()
                n = features[n_indices[dist[index][n_indices].argmin()]]
                positives.append(p)
                negatives.append(n)
        else:
            for mask in labels_mask:
                p = features[choice(nonzero(mask).flatten())]
                n = features[choice(nonzero(logical_not(mask)).flatten())]
                positives.append(p)
                negatives.append(n)
        return self.__base__(features, stack(positives), stack(negatives))
