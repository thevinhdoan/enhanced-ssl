import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, lambda_1: float, lambda_2: float):
        super(Loss, self).__init__()
        self.lambda_1 = lambda_1 # weighting for similarity-to-centroid (alignment) term
        self.lambda_2 = lambda_2 # weighting for similarity-to-others (robustness) term
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, train_logits:torch.Tensor, train_targets:torch.Tensor, unlabeled_logits:torch.Tensor, batch_grouping:dict):
        """
        - train_logits: Tensor of shape (num_train_samples, num_classes) - model predictions on the train set.
        - train_targets: Tensor of shape (num_train_samples,) - ground truth labels for the train set.
        - unlabeled_logits: Tensor of shape (num_unlabeled_samples, num_classes) - model predictions on the unlabeled set.
        - batch_grouping: same format as the original grouping, but adapted to use indices within train_logits and unlabeled_logits.
        """
        num_classes = train_logits.shape[1]
        unlabeled_size = unlabeled_logits.shape[0]
        assert train_logits.shape[0] == train_targets.shape[0], \
            "The number of predictions must match the number of targets."
        assert unlabeled_logits.shape[1] == num_classes, \
            "Predictions on the train/unlabeled sets must have the same number of classes."
        # assert sum(len(x) for x in batch_grouping.values()) == unlabeled_size, \
        #     f"The sum of the lengths of the batch_grouping values must equal the size of unlabeled_logits.\nBatch grouping: {batch_grouping}\nUnlabeled size: {unlabeled_size}"

        # 1. Standard cross-entropy loss on the train set
        cross_entropy_term = self.cross_entropy_loss(train_logits, train_targets)

        # 2. Alignment term: encourage unlabeled samples to be close to their cluster centroids
        alignment_term = 0
        for centroid_id, member_ids in batch_grouping.items():
            if centroid_id < 0 or len(member_ids) == 0:
                continue
            centroid_probs = train_logits[centroid_id]
            centroid_probs = centroid_probs.unsqueeze(0)
            centroid_probs = centroid_probs.expand(len(member_ids), num_classes)
            member_probs = unlabeled_logits[member_ids]
            in_cluster_distance = torch.norm(centroid_probs - member_probs, dim=1, p=2).sum()
            alignment_term += (in_cluster_distance / unlabeled_size)

        # 3. Robustness term: encourage unlabeled samples to be close to each other within the same cluster
        robustness_term = 0
        for member_ids in batch_grouping.values():
            if len(member_ids) <= 1:
                continue
            member_probs = unlabeled_logits[member_ids]
            cluster_size = len(member_ids)
            pairwise_distance = 0
            for i in range(cluster_size):
                for j in range(i + 1, cluster_size):
                    pairwise_distance += torch.norm(member_probs[i] - member_probs[j], p=2)
            robustness_term += (pairwise_distance / (cluster_size * unlabeled_size))

        # Final loss, with weighting factors
        return cross_entropy_term + self.lambda_1 * alignment_term + self.lambda_2 * robustness_term
