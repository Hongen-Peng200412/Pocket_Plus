import os
import torch
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="sparse_h100")
def main(cfg: DictConfig):
    print("Instantiating dataset and model...")
    # Instantiate the wrapper
    try:
        wrapper = hydra.utils.instantiate(cfg.model)
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    # In BoxPointDataset, voxel_grid usually has e.g. 14 channels
    in_channels = 14
    wrapper.backbone.set_input_channels(in_channels)
    
    print("Model instantiated successfully.")
    
    # Mock data
    B = 2
    N_list = [100, 150]
    sumN = sum(N_list)
    D, H, W = 32, 32, 32
    
    # PTV3 offsets
    offsets = torch.tensor([100, 250], dtype=torch.long)
    batch_index = torch.cat([torch.full((N_list[0],), 0, dtype=torch.long), 
                             torch.full((N_list[1],), 1, dtype=torch.long)])
    
    # Check atom_feature_dim
    atom_feat_dim = cfg.model.backbone.point_backbone.atom_feature_dim
    
    batch = {
        "voxel_grid": torch.randn(B, in_channels, D, H, W, dtype=torch.float32),
        "atom_feat": torch.randn(sumN, atom_feat_dim, dtype=torch.float32),
        "atom_coord_centered_world": torch.randn(sumN, 3, dtype=torch.float32),
        "atom_batch_index": batch_index,
        "atom_offsets": offsets,
        "atom_coord_local_voxel": torch.rand(sumN, 3, dtype=torch.float32) * min(D, H, W),
        "box_shape_zyx": torch.tensor([[D, H, W], [D, H, W]], dtype=torch.float32),
        "voxel_size_world": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        "atom_is_in_core_box": torch.ones(sumN, dtype=torch.bool),
        "atom_label": torch.randint(0, 2, (sumN,), dtype=torch.float32),
        "atom_valid_mask": torch.ones(sumN, dtype=torch.bool),
        "voxel_label": torch.randint(0, 2, (B, D, H, W), dtype=torch.float32),
        "voxel_valid_mask": torch.ones(B, D, H, W, dtype=torch.bool),
        "hardmask": torch.ones(B, D, H, W, dtype=torch.bool),
    }

    print("Running forward pass...")
    try:
        # Move to GPU if available
        if torch.cuda.is_available():
            wrapper = wrapper.cuda()
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
        outputs = wrapper(batch)
        print("Forward pass successful.")
        
        print("Computing loss...")
        loss_dict = wrapper._compute_total_loss(outputs, batch)
        print(f"Loss computed successfully: {loss_dict[1]}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
