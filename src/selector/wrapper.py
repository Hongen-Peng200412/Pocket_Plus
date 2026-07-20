"""CCLN 的损失装配、按 CLG 等权的小批次平均和配置构造入口。

`SelectorWrapper` 不改变 CCLN 的预测结构，只把 CLG 有效性、候选最大 IoU 回归和
非空反链条件似然三项监督组合成训练损失。反链项只在在线最优监督为非空时定义；
`oracle/detached/joint` 三种模式仅改变该项乘以门控概率时的梯度路径。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model.ccln import CandidateConditionedLineageNetwork
from .structured.antichain_dp import exact_antichain_log_partition


def binary_focal_loss_with_logits(
    logit: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    计算未归约二分类 focal loss；``gamma=0`` 时严格等于 BCE-with-logits。

    输入参数:
        - logit: torch.Tensor, 任意 shape，未 sigmoid 的预测值
        - target: torch.Tensor, 与 `logit` 同 shape，元素取值为 0/1
        - gamma: float, focal 聚焦参数

    输出:
        - loss: torch.Tensor, 与 `logit` 同 shape，尚未归约的 focal loss
    """
    bce = F.binary_cross_entropy_with_logits(logit, target.to(dtype=logit.dtype), reduction="none")
    if float(gamma) == 0.0:
        return bce
    probability = torch.sigmoid(logit)
    probability_true = torch.where(target.to(dtype=torch.bool), probability, 1.0 - probability)
    return (1.0 - probability_true).pow(float(gamma)) * bce


class SelectorWrapper(nn.Module):
    """
    组合 CCLN、CLG/blob/反链 loss 与三种条件加权方式。

    输入参数:
        - model: CandidateConditionedLineageNetwork, 首版唯一 Selector 网络
        - lambda_count: float, 预测/oracle 反链计数惩罚
        - condition_weighting: str, `oracle`、`detached` 或 `joint`
        - gamma_focal: float, L_CLG focal gamma；0 表示 BCE
        - w_clg/w_blob/w_antichain: float, 三项 loss 权重
        - smooth_l1_beta: float, qhat 对 q 的 SmoothL1 beta

    前向输入:
        - samples: Sequence[Mapping[str,Any]], 长度 B，每项一个完整 CLG

    前向输出:
        - outputs: list[dict[str,torch.Tensor]], 长度 B，与输入 CLG 逐项对齐
    """

    def __init__(
        self,
        model: CandidateConditionedLineageNetwork,
        lambda_count: float,
        condition_weighting: str,
        gamma_focal: float,
        w_clg: float,
        w_blob: float,
        w_antichain: float,
        smooth_l1_beta: float,
    ) -> None:
        """
        保存 CCLN、结构化惩罚和三项损失的冻结配置。

        输入参数:
            - model: CandidateConditionedLineageNetwork, 被训练和保存的完整 CCLN
            - lambda_count: float, 预测与在线最优监督共用的候选计数惩罚
            - condition_weighting: str, `oracle/detached/joint` 之一
            - gamma_focal: float, CLG 二分类 focal 聚焦参数；0 等价于 BCE
            - w_clg: float, CLG 有效性损失权重
            - w_blob: float, 候选最大 IoU 回归损失权重
            - w_antichain: float, 加权反链条件负对数似然权重
            - smooth_l1_beta: float, 候选最大 IoU SmoothL1 损失的二次区间宽度

        条件权重:
            - `oracle`: 正 CLG 的反链项权重恒为 1。
            - `detached`: 权重为 `p_G`，但不让反链损失梯度进入 CLG 门控分支。
            - `joint`: 权重为 `p_G`，并保留到 CLG 门控分支的梯度。
        """
        super().__init__()
        if condition_weighting not in {"oracle", "detached", "joint"}:
            raise ValueError("condition_weighting 只允许 oracle、detached、joint。")
        self.model = model
        self.lambda_count = float(lambda_count)
        self.condition_weighting = str(condition_weighting)
        self.gamma_focal = float(gamma_focal)
        self.w_clg = float(w_clg)
        self.w_blob = float(w_blob)
        self.w_antichain = float(w_antichain)
        self.smooth_l1_beta = float(smooth_l1_beta)

    def forward(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, torch.Tensor]]:
        """
        逐 CLG 执行 CCLN，保留每个样本自己的 ragged/tree 结构。

        输入参数:
            - samples: Sequence[Mapping[str,Any]], 长度 B，每项一个 CLG

        输出:
            - outputs: list[dict[str,torch.Tensor]], 长度 B，每项含 qhat/z/a_G/p_G
        """
        return [self.model(sample) for sample in samples]

    def compute_loss(
        self,
        samples: Sequence[Mapping[str, Any]],
        outputs: Sequence[Mapping[str, torch.Tensor]] | None,
    ) -> dict[str, torch.Tensor]:
        """
        先在每个 CLG 内归约损失，再对小批次中的 CLG 等权平均。

        输入参数:
            - samples: Sequence[Mapping[str, Any]], 长度 B，必须包含在线最优监督字段
            - outputs: Sequence[Mapping[str, torch.Tensor]] | None, 已有前向结果；
              None 表示由本函数现场调用 CCLN

        输出:
            - losses: dict[str, torch.Tensor]，包含:
                - "total_loss": scalar，三个权重项的 batch 平均总损失
                - "CLG_loss": scalar，a_G 对 y_G 的 BCE/focal
                - "blob_loss": scalar，全部正/负 CLG 的逐 CLG SmoothL1 均值
                - "antichain_loss": scalar，正 CLG 非空条件分布负对数似然的未门控均值
                - "weighted_antichain_loss": scalar，按 `oracle/detached/joint`
                  条件权重处理后的反链项均值
                - "mean_CLG_valid_probability": scalar，当前 batch 平均 p_G
        """
        prediction = self(samples) if outputs is None else list(outputs)
        if len(prediction) != len(samples) or not samples:
            raise ValueError("samples 与 outputs 必须为相同非零长度。")

        # 六个列表均逐 CLG 保存 scalar；函数末尾才在小批次维做等权平均。
        clg_losses: list[torch.Tensor] = []
        blob_losses: list[torch.Tensor] = []
        antichain_losses: list[torch.Tensor] = []
        weighted_antichain_losses: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        total_losses: list[torch.Tensor] = []
        for sample, output in zip(samples, prediction, strict=True):
            target_valid = sample["CLG_is_valid"].reshape(()).to(
                device=output["CLG_logit"].device,
                dtype=output["CLG_logit"].dtype,
            )
            clg_loss = binary_focal_loss_with_logits(
                output["CLG_logit"].reshape(()), target_valid, self.gamma_focal
            )
            target_q = sample["candidate_max_iou"].to(
                device=output["predicted_max_iou"].device,
                dtype=output["predicted_max_iou"].dtype,
            )
            # 先在本 CLG candidates 内平均；之后各 CLG 在 batch 中权重相同。
            blob_loss = F.smooth_l1_loss(
                output["predicted_max_iou"],
                target_q,
                reduction="mean",
                beta=self.smooth_l1_beta,
            )

            if bool(target_valid.detach().item()):
                # log Z 遍历全部非空反链；oracle_energy 只取在线最优监督反链。
                log_partition = exact_antichain_log_partition(
                    selection_logit=output["selection_logit"],
                    parent_index=sample["closure_parent_index"],
                    candidate_index_by_node=sample["closure_candidate_index_by_node"],
                    lambda_count=self.lambda_count,
                )
                oracle_selected = sample["oracle_selected_candidate_index"].to(
                    device=output["selection_logit"].device,
                    dtype=torch.long,
                )
                oracle_energy = output["selection_logit"][oracle_selected].sum() - (
                    self.lambda_count * oracle_selected.numel()
                )
                antichain_loss = log_partition - oracle_energy
                if self.condition_weighting == "oracle":
                    condition_weight = output["CLG_valid_probability"].new_ones(())
                elif self.condition_weighting == "detached":
                    condition_weight = output["CLG_valid_probability"].detach()
                else:
                    condition_weight = output["CLG_valid_probability"]
            else:
                antichain_loss = output["selection_logit"].sum() * 0.0
                condition_weight = output["CLG_valid_probability"].new_zeros(())
            weighted_antichain = condition_weight * antichain_loss

            # L_blob 对正、负 CLG 均始终训练；三种条件权重只作用于正 CLG 的 L_antichain。
            total = (
                self.w_clg * clg_loss
                + self.w_blob * blob_loss
                + self.w_antichain * weighted_antichain
            )
            clg_losses.append(clg_loss)
            blob_losses.append(blob_loss)
            antichain_losses.append(antichain_loss)
            weighted_antichain_losses.append(weighted_antichain)
            probabilities.append(output["CLG_valid_probability"])
            total_losses.append(total)

        return {
            "total_loss": torch.stack(total_losses).mean(),
            "CLG_loss": torch.stack(clg_losses).mean(),
            "blob_loss": torch.stack(blob_losses).mean(),
            "antichain_loss": torch.stack(antichain_losses).mean(),
            "weighted_antichain_loss": torch.stack(weighted_antichain_losses).mean(),
            "mean_CLG_valid_probability": torch.stack(probabilities).mean(),
        }


def build_selector_wrapper_from_config(
    config: Mapping[str, Any],
    source_dimensions: Mapping[str, Mapping[str, int]],
) -> SelectorWrapper:
    """
    从已解析配置和真实产物来源通道数实例化完整 CCLN 包装器。

    输入参数:
        - config: Mapping[str, Any], 完整 Selector 配置，必须含 `model/loss/data`
          三组
        - source_dimensions: Mapping[str, Mapping[str, int]], Dataset 首个样本解析
          出的 A/P 字段末维通道数

    输出:
        - wrapper: SelectorWrapper，严格按配置来源顺序构造的可训练 CCLN 和损失装配；
          未启用模态不会创建对应融合参数
    """
    # 三组配置分别控制网络结构、损失组合和在线最优监督计数惩罚。
    model_config = config["model"]
    loss_config = config["loss"]
    data_config = config["data"]
    a_source_order = tuple(str(value) for value in model_config["A_source_order"])
    p_source_order = tuple(str(value) for value in model_config["P_source_order"])
    a_source_dims = {name: int(source_dimensions["A"][name]) for name in a_source_order}
    p_source_dims = {name: int(source_dimensions["P"][name]) for name in p_source_order}
    network = CandidateConditionedLineageNetwork(
        modalities=model_config["modalities"],
        a_source_order=a_source_order,
        a_source_dims=a_source_dims,
        p_source_order=p_source_order,
        p_source_dims=p_source_dims,
        hidden_dim=int(model_config["hidden_dim"]),
        num_heads=int(model_config["num_heads"]),
        tree_layers=int(model_config["tree_layers"]),
        tree_ffn_hidden_dim=int(model_config["tree_ffn_hidden_dim"]),
        candidate_attribute_dim=int(model_config["candidate_attribute_dim"]),
        modality_projected_dim=int(model_config["modality_projected_dim"]),
        modality_gate_hidden_dim=int(model_config["modality_gate_hidden_dim"]),
        v_output_dim=int(model_config["V_output_dim"]),
        v_meta_dim=int(model_config["V_meta_dim"]),
        v_correction_gate_hidden_dim=int(model_config["V_correction_gate_hidden_dim"]),
        use_density_context=bool(model_config["use_density_context"]),
        density_input_shape_zyx=model_config["density_input_shape_zyx"],
        density_channels=model_config["density_channels"],
        density_bottleneck_heads=int(model_config["density_bottleneck_heads"]),
        density_bottleneck_layers=int(model_config["density_bottleneck_layers"]),
        density_bottleneck_ffn_dim=int(model_config["density_bottleneck_ffn_dim"]),
        dropout=float(model_config["dropout"]),
        probability_epsilon=float(model_config["probability_epsilon"]),
    )
    return SelectorWrapper(
        model=network,
        lambda_count=float(data_config["lambda_count"]),
        condition_weighting=str(loss_config["condition_weighting"]),
        gamma_focal=float(loss_config["gamma_focal"]),
        w_clg=float(loss_config["w_CLG"]),
        w_blob=float(loss_config["w_blob"]),
        w_antichain=float(loss_config["w_antichain"]),
        smooth_l1_beta=float(loss_config["smooth_l1_beta"]),
    )
