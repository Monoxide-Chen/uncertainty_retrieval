import random

import torch
import numpy as np
import wandb

from utils.metrics import AverageMeterSet


class ValidationMetricsCalculator:
    def __init__(self, original_query_features: torch.tensor, composed_query_features: torch.tensor,
                 test_features: torch.tensor, attribute_matching_matrix: np.array,
                 ref_attribute_matching_matrix: np.array, top_k: tuple):
        self.original_query_features = original_query_features
        self.composed_query_features = composed_query_features
        self.test_features = test_features
        self.top_k = top_k
        self.attribute_matching_matrix = attribute_matching_matrix
        self.ref_attribute_matching_matrix = ref_attribute_matching_matrix
        self.num_query_features = composed_query_features.size(0)
        self.num_test_features = test_features.size(0)
        self.similarity_matrix = torch.zeros(self.num_query_features, self.num_test_features)
        self.top_scores = torch.zeros(self.num_query_features, max(top_k))
        self.most_similar_idx = torch.zeros(self.num_query_features, max(top_k))
        self.recall_results = {}
        self.recall_positive_queries_idxs = {k: [] for k in top_k}
        self.similarity_matrix_calculated = False
        self.top_scores_calculated = False

    def __call__(self):
        self._calculate_similarity_matrix()
        # Filter query_feat == target_feat
        assert self.similarity_matrix.shape == self.ref_attribute_matching_matrix.shape
        self.similarity_matrix[self.ref_attribute_matching_matrix == True] = self.similarity_matrix.min()
        return self._calculate_recall_at_k()

    def _calculate_similarity_matrix(self) -> torch.tensor:
        """
        query_features = torch.tensor. Size = (N_test_query, Embed_size)
        test_features = torch.tensor. Size = (N_test_dataset, Embed_size)
        output = torch.tensor, similarity matrix. Size = (N_test_query, N_test_dataset)
        """
        if not self.similarity_matrix_calculated:
            self.similarity_matrix = self.composed_query_features.mm(self.test_features.t())
            self.similarity_matrix_calculated = True

    def _calculate_recall_at_k(self):
        average_meter_set = AverageMeterSet()
        self.top_scores, self.most_similar_idx = self.similarity_matrix.topk(max(self.top_k))
        self.top_scores_calculated = True
        topk_attribute_matching = np.take_along_axis(self.attribute_matching_matrix, self.most_similar_idx.numpy(),
                                                     axis=1)

        for k in self.top_k:
            query_matched_vector = topk_attribute_matching[:, :k].sum(axis=1).astype(bool)
            self.recall_positive_queries_idxs[k] = list(np.where(query_matched_vector > 0)[0])
            num_correct = query_matched_vector.sum()
            num_samples = len(query_matched_vector)
            average_meter_set.update('recall_@{}'.format(k), num_correct, n=num_samples)
        recall_results = average_meter_set.averages()
        return recall_results

    @staticmethod
    def _multiple_index_from_attribute_list(attribute_list, indices):
        attributes = []
        for idx in indices:
            attributes.append(attribute_list[idx.item()])
        return attributes
