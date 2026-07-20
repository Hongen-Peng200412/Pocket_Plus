"""CCLN 的模态边界、10D bias 与内容/树读出分离测试. """

from __future__ import annotations

import torch

from src.selector.model.ccln import CandidateConditionedLineageNetwork
from src.selector.wrapper import SelectorWrapper


def _build_v_only_model() -> CandidateConditionedLineageNetwork:
    """构造使用小网格但保持同一 CCLN 结构的 V-only 模型. """
    return CandidateConditionedLineageNetwork(
        modalities=("V",),
        a_source_order=(),
        a_source_dims={},
        p_source_order=(),
        p_source_dims={},
        hidden_dim=32,
        num_heads=4,
        tree_layers=2,
        tree_ffn_hidden_dim=64,
        candidate_attribute_dim=16,
        modality_projected_dim=16,
        modality_gate_hidden_dim=32,
        v_output_dim=48,
        v_meta_dim=2,
        v_correction_gate_hidden_dim=16,
        use_density_context=True,
        density_input_shape_zyx=(16, 16, 16),
        density_channels=(4, 8, 8, 16),
        density_bottleneck_heads=4,
        density_bottleneck_layers=1,
        density_bottleneck_ffn_dim=32,
        dropout=0.0,
        probability_epsilon=1e-6,
    )


def _v_only_sample() -> dict:
    """构造两个 candidates、三个 V voxels 的轻量 CLG. """
    coord_world = torch.tensor([[1.5, 1.5, 1.5], [2.5, 2.5, 2.5], [3.5, 3.5, 3.5]])
    return {
        "candidate_attributes": torch.randn(2, 16),
        "candidate_centroid_world": torch.tensor([[2.0, 2.0, 2.0], [3.5, 3.5, 3.5]]),
        "tree_relative_feature": torch.randn(2, 2, 8),
        "box_center_world": torch.tensor([8.0, 8.0, 8.0]),
        "box_half_size_world": torch.tensor([8.0, 8.0, 8.0]),
        "box_shape_zyx": torch.tensor([16, 16, 16]),
        "oldest_V_center_world": coord_world.mean(dim=0),
        "candidate_voxel_offsets": torch.tensor([0, 2, 3]),
        "candidate_voxel_index": torch.tensor([0, 1, 2]),
        "V_sources": {"voxel_final": torch.randn(3, 48)},
        "V_probability": torch.rand(3),
        "V_coord_world": coord_world,
        "V_index_local_zyx": torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        "V_batch_index": torch.zeros((3,), dtype=torch.long),
        "density_input": torch.randn(1, 1, 16, 16, 16),
    }


def test_ccln_has_zero_initialized_probability_prior_and_expected_outputs() -> None:
    """每个模态概率先验应从 0 开始, 输出字段应区分 qhat/z/a_G. """
    model = _build_v_only_model().eval()
    assert float(model.modality_attention["V"].probability_prior.detach()) == 0.0
    with torch.no_grad():
        output = model(_v_only_sample())
    assert output["candidate_content"].shape == (2, 32)
    assert output["candidate_tree"].shape == (2, 32)
    assert output["predicted_max_iou"].shape == (2,)
    assert bool(torch.all((0.0 <= output["predicted_max_iou"]) & (output["predicted_max_iou"] <= 1.0)))
    assert output["selection_logit"].shape == (2,)
    assert output["CLG_logit"].shape == ()


def test_blob_readout_is_upstream_of_tree_transformer() -> None:
    """改变 tree Transformer 参数不得改变同一输入的 E_content/qhat. """
    torch.manual_seed(7)
    model = _build_v_only_model().eval()
    sample = _v_only_sample()
    with torch.no_grad():
        before = model(sample)
        for layer in model.tree_layers:
            for parameter in layer.parameters():
                parameter.add_(torch.randn_like(parameter) * 3.0)
        after = model(sample)
    torch.testing.assert_close(before["candidate_content"], after["candidate_content"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(before["predicted_max_iou"], after["predicted_max_iou"], rtol=0.0, atol=0.0)
    assert not torch.equal(before["candidate_tree"], after["candidate_tree"])


def test_find_empty_p_and_a_use_learned_null_without_fake_disk_tokens() -> None:
    """Find BOX 内空 P/A 子集应由模型 null token 处理, 并允许真实表长度为 0. """
    model = CandidateConditionedLineageNetwork(
        modalities=("V", "P", "A"),
        a_source_order=("L0", "L1"),
        a_source_dims={"L0": 3, "L1": 5},
        p_source_order=("L2",),
        p_source_dims={"L2": 4},
        hidden_dim=32,
        num_heads=4,
        tree_layers=1,
        tree_ffn_hidden_dim=64,
        candidate_attribute_dim=16,
        modality_projected_dim=8,
        modality_gate_hidden_dim=16,
        v_output_dim=48,
        v_meta_dim=2,
        v_correction_gate_hidden_dim=16,
        use_density_context=False,
        density_input_shape_zyx=(16, 16, 16),
        density_channels=(4, 8, 8, 16),
        density_bottleneck_heads=4,
        density_bottleneck_layers=1,
        density_bottleneck_ffn_dim=32,
        dropout=0.0,
        probability_epsilon=1e-6,
    ).eval()
    sample = _v_only_sample()
    sample["density_input"] = None
    sample["P_sources"] = {"L2": torch.empty((0, 4))}
    sample["P_probability"] = torch.empty((0,))
    sample["P_coord_world"] = torch.empty((0, 3))
    sample["A_sources"] = {"L0": torch.empty((0, 3)), "L1": torch.empty((0, 5))}
    sample["A_probability"] = torch.empty((0,))
    sample["A_coord_world"] = torch.empty((0, 3))
    sample["oldest_A_center_world"] = sample["box_center_world"]
    sample["candidate_A_offsets"] = torch.tensor([0, 0, 0])
    sample["candidate_A_index"] = torch.empty((0,), dtype=torch.long)
    with torch.no_grad():
        output = model(sample)
    assert output["selection_logit"].shape == (2,)

    sample["P_sources"] = {"L2": torch.randn(2, 4)}
    sample["P_probability"] = torch.tensor([0.3, 0.7])
    sample["P_coord_world"] = torch.tensor([[2.0, 1.0, 1.0], [4.0, 3.0, 2.0]])
    sample["A_sources"] = {"L0": torch.randn(3, 3), "L1": torch.randn(3, 5)}
    sample["A_probability"] = torch.tensor([0.2, 0.6, 0.8])
    sample["A_coord_world"] = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    )
    sample["oldest_A_center_world"] = sample["A_coord_world"].mean(dim=0)
    sample["candidate_A_offsets"] = torch.tensor([0, 2, 3])
    sample["candidate_A_index"] = torch.tensor([0, 1, 2])
    with torch.no_grad():
        nonempty_output = model(sample)
    assert nonempty_output["selection_logit"].shape == (2,)


def test_v48_fallback_ignores_density_artifact() -> None:
    """关闭密度分支后模型只消费 V48 基础表。"""
    model = CandidateConditionedLineageNetwork(
        modalities=("V",),
        a_source_order=(),
        a_source_dims={},
        p_source_order=(),
        p_source_dims={},
        hidden_dim=32,
        num_heads=4,
        tree_layers=1,
        tree_ffn_hidden_dim=64,
        candidate_attribute_dim=16,
        modality_projected_dim=8,
        modality_gate_hidden_dim=16,
        v_output_dim=48,
        v_meta_dim=2,
        v_correction_gate_hidden_dim=16,
        use_density_context=False,
        density_input_shape_zyx=(16, 16, 16),
        density_channels=(4, 8, 8, 16),
        density_bottleneck_heads=4,
        density_bottleneck_layers=1,
        density_bottleneck_ffn_dim=32,
        dropout=0.0,
        probability_epsilon=1e-6,
    ).eval()
    sample = _v_only_sample()
    sample["density_input"] = None
    with torch.no_grad():
        output = model(sample)
    assert output["predicted_max_iou"].shape == (2,)


def test_ccln_wrapper_backward_reaches_density_blob_tree_and_clg_paths() -> None:
    """一个真实三项 loss 反传应同时覆盖密度、qhat、反链与 CLG readout. """
    model = _build_v_only_model().train()
    wrapper = SelectorWrapper(
        model=model,
        lambda_count=0.05,
        condition_weighting="oracle",
        gamma_focal=0.0,
        w_clg=1.0,
        w_blob=1.0,
        w_antichain=1.0,
        smooth_l1_beta=1.0,
    )
    sample = _v_only_sample()
    sample.update(
        {
            "candidate_max_iou": torch.tensor([0.2, 0.8]),
            "oracle_selected_candidate_index": torch.tensor([1]),
            "CLG_is_valid": torch.tensor(1.0),
            "closure_parent_index": (-1, 0),
            "closure_candidate_index_by_node": (0, 1),
        }
    )
    losses = wrapper.compute_loss([sample], None)
    losses["total_loss"].backward()
    density_encoder = model.v_fusion.density_encoder
    assert density_encoder is not None
    assert density_encoder.input_projection.weight.grad is not None
    assert model.blob_head[-1].weight.grad is not None
    assert model.selection_head[-1].weight.grad is not None
    assert model.clg_head[-1].weight.grad is not None
