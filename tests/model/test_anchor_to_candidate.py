from __future__ import annotations

import torch

from src.model.sparse_refine.interpolation import AnchorToCandidateKnnSearch


def test_knn_message_uses_same_box_and_route_class() -> None:
    """
    验证邻居搜索只连接同 BOX 且同路由类别的 P，并自然 padding 不足的邻居。
    """
    search = AnchorToCandidateKnnSearch("knn_message", num_neighbors=3, same_class_only=True, chunk_size=1)
    output = search(
        candidate_coord_centered_world=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        candidate_batch_index=torch.tensor([0, 0, 0]),
        candidate_class=torch.tensor([1, 2, 3]),
        anchor_coord_centered_world=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        anchor_batch_index=torch.tensor([0, 0, 0]),
        anchor_class=torch.tensor([1, 1, 2]),
    )

    assert output["candidate_neighbor_valid_mask"].tolist() == [[True, True, False], [True, False, False], [False, False, False]]
    torch.testing.assert_close(output["candidate_neighbor_squared_distance"][0, :2], torch.tensor([0.0, 4.0]))
    torch.testing.assert_close(output["candidate_neighbor_relative_coords"][1, 0], torch.zeros(3))


def test_knn_message_chunked_results_match_full_chunk() -> None:
    """
    验证分块查询不改变 KNN 邻居结果。
    """
    kwargs = {
        "candidate_coord_centered_world": torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        "candidate_batch_index": torch.zeros(3, dtype=torch.long),
        "candidate_class": torch.ones(3, dtype=torch.long),
        "anchor_coord_centered_world": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        "anchor_batch_index": torch.zeros(3, dtype=torch.long),
        "anchor_class": torch.ones(3, dtype=torch.long),
    }
    chunked = AnchorToCandidateKnnSearch("knn_message", 2, True, 1)(**kwargs)
    full = AnchorToCandidateKnnSearch("knn_message", 2, True, 32)(**kwargs)

    assert torch.equal(chunked["candidate_neighbor_index"], full["candidate_neighbor_index"])
    torch.testing.assert_close(chunked["candidate_neighbor_squared_distance"], full["candidate_neighbor_squared_distance"])
