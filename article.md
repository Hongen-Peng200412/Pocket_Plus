# Methods

## Step 1: Definition of Ligands and Pockets
This study adopts a broad definition for "ligands". Specifically, we first enumerate almost all `HETATM` records from the structure files, and then filter out non-biological components using explicit exclusion rules and geometric contact criteria. Unlike databases that focus solely on druggable small molecules, our ligands encompass not only conventional small molecules but also metal ions, short peptides, and nucleic acid ligands. However, compared to retaining all `HETATM` records indiscriminately, we proactively exclude meaningless entries (e.g., water molecules, common buffers/crystallization additives, and dummy atoms). This aims to cover biologically relevant ligands that genuinely participate in interactions and may leave local signals in cryo-EM density maps.

Specifically: for each structure, we parse all protein and nucleic acid receptor heavy atoms and extract candidate ligand residues from `HETATM` records. Water molecules, common non-specific solvents, and buffer system molecules (e.g., glycerol, PEG, TRIS, DMSO) are permanently excluded. Furthermore, if a modified residue is covalently linked to the protein or nucleic acid backbone, it is treated as part of the receptor rather than an independent ligand.

Following BioLiP2, we apply the following constraints: for non-hydrogen atoms, if the distance between a receptor atom and a ligand atom is ≤ 4.0/5.0 Å, it is recorded as an intermolecular atomic contact. If a receptor residue forms at least 2 such contacts with the same ligand, it is defined as a binding residue; ligands corresponding to less than 2 binding residues are excluded.

In our code implementation, a key feature is that the ligand filtering rules are abstracted into two independent steps: attribute extraction and attribute filtering. The codebase supports direct extraction of attributes such as ligand category, atom count, contact heavy atom count, and contact residue count. Users can easily declare rules in the configuration (e.g., contact heavy atoms > 5, exclude all metal ions), and the program automatically extracts ligands, assigns labels, and performs training and inference.

The current supervised learning task is formulated as binary semantic segmentation. That is, if any atom or voxel belongs to a binding pocket induced by any defined ligand, it is uniformly labeled as positive.

## Step 2: Feature Extraction from PDB and Cryo-EM Density Maps
For each structure, we parse the receptor's heavy atom set from PDB/mmCIF files, ignoring hydrogen atoms. For each receptor atom, we construct a 49-dimensional feature vector encompassing its elemental identity (6-dim one-hot), residue type (25-dim one-hot, including standard amino acids, nucleotides, and mapping modified residues to their parents), physicochemical properties (8-dim for polarity, acidity/basicity, and charge), normalized atomic mass (1-dim), and local density histogram features (9-dim).

In addition to chemical features, we preserve the exact 3D coordinates. During training, three coordinate representations are synchronized for each BOX: raw world coordinates, continuous local voxel coordinates relative to the BOX origin, and centered world coordinates relative to the BOX center. These play an important role during learning.

For cryo-EM density maps, we organize the arrays into a 3D grid with the order (Z, Y, X) and resample to 1.0 Å. The experimental density maps undergo a global Z-score normalization. Furthermore, we construct difference maps to explicitly manifest ligand densities by subtracting the simulated density map of the ligand-free receptor from the experimental map. Various derivatives of the difference map (e.g., positive residual maps, negative residual maps, normalized, and multi-scale Gaussian smoothed versions) are introduced as additional voxel channels to enhance the model's sensitivity to ligand-specific residual signals.

To align structural information with density maps, atomic features and labels are projected onto the resampled EM grid using their continuous local voxel coordinates, thereby generating corresponding feature and label grids.

## Step 3: Preprocessing of Training Data
Our training data comes from two sources. First, we generate training BOXes around each identified pocket via sliding windows over a candidate bounding box, accepting BOXes only when the positive voxel count and centeredness exceed specific thresholds. This filters out weakly relevant background regions. Second, we apply random cropping, accepting samples as long as the total atom count exceeds a certain proportion.

Currently, a default $80 \times 80 \times 80$ voxel window is used. We apply simultaneous 90° random 3D rotations to both voxels and atoms during training. For each BOX, receptor atoms within the box are defined as "core atoms", while those outside the box but within a buffer radius are "buffer atoms". Buffer atoms only provide structural context and are excluded from the loss calculation.

We adopt a two-stage training curriculum: the model is first trained exclusively on the first type of dataset to learn foundational pocket features, then the second type is introduced until final convergence.

## Step 4: Network Framework
Our network framework comprises four main modules—an `embed head`, a `voxel backbone`, a `point backbone`, and an `atom head`—and integrates three core mechanisms: pseudo-atoms, cross-modal information fusion, and a recycle mechanism.

Given a density patch $V$ cut into a BOX and its corresponding atom set $P$, the embed head extracts receptor information and projects the atoms into an intermediate embedded representation. Subsequently, the voxel backbone processes the density grid alongside the embedded features, while the point backbone injects both real and pseudo-atoms, processing them using the embedded features and atomic properties. During forward propagation, the point backbone receives density context from the voxel backbone to perform cross-modal fusion, and ultimately, the atom head directly outputs the probability of each atom belonging to a pocket.

### Modules:
**1. Embed Head:**
When processing atoms, the embed head utilizes a shared trunk encoding and a progressive cropping (渐进裁剪) mechanism. Its initial input is not strictly limited to atoms within the core BOX; rather, it includes buffer atoms. As the network processes features hierarchically, it trims off layers of buffer atoms progressively without computing hard multi-stage interactions immediately. By the time features exit the embed head, only the core atoms remain. It outputs enriched point representations (which add as residuals to the point backbone features) and voxelized structural embeddings that serve as auxiliary density channels for the voxel backbone. This avoids exhausting precious volumetric memory while preserving an extended receptive field.

**2. Voxel Backbone:**
Based on a 3D U-Net (RAUNet) architecture, the voxel branch maintains a stable density representation. It incorporates 3D Rotary Position Embedding (RoPE) self-attention in the bottleneck and attention gates in the decoder. The voxel branch explicitly exports multi-scale intermediate voxel features for voxel-to-point fusion, while returning an auxiliary voxel segmentation head to retain learning stability over the volumetric domain.

**3. Point Backbone:**
The point cloud branch is built on a lightweight version of Point Transformer V3 (PTV3). Since our task is atom-level binary classification without the need for intensive sparse decoding, we replaced the native sparse convolutions with a lightweight PointConv-style conditional position encoding (CPE) to avoid severe sparsity issues and improve parameter efficiency. The point branch employs space-filling curves (e.g., Z-order, Hilbert) to serialize 3D points, transforming the unorganized point cloud into a 1D sequence for efficient serialized attention. We also employ a Gated Feed-Forward Network (Gated FFN) to replace the standard FFN, bolstering representation capacity. During its forward pass, the point backbone incorporates point recycle states as residual inputs from the previous recycle iterations and leaves hooking spots for multi-scale density incorporations.

**4. Atom Head:**
To complete the atom-level pocket discrimination, we appended an independent atom head following the point backbone. Instead of a simple linear classifier, the atom head concatenates the fused point features, geometric attributes, and valid masks to create unified atomic tokens. These run through a `Stage1SerializedAttentionStack` to capture structural dependencies before projecting to the final binary logits via an MLP module.

### Mechanisms:
**1. Pseudo-Atom Mechanism:**
Driven by the prior assumption that regions with prominent positive differences (experimental map - simulated map) are highly probable ligand locations, we dynamically generate pseudo-atoms during the forward pass. Candidate pseudo-atoms are sampled based on the density probability derived from difference map channels. We prune candidates strictly using a set of filters—removing ones too close to or too far from real receptor atoms—and deduplicate the rest via a greedy spatial clustering sequence to avoid noise drift. These pseudo-atoms act purely as auxiliary message-passing nodes inside the point backbone to enhance density-driven contextual relationships; they are excluded from the loss calculation (`valid_mask = False`) and scrubbed off safely right before the real atoms transition into the atom head.

**2. Information Fusion Mechanism:**
We implement a uni-directional voxel-to-point fusion strategy, averting explicit point-to-voxel back-writing to streamline computational efficiency. Whenever an integrated fusion stage arises along the point backbone sequence, continuous local coordinates of points are computed, querying corresponding localized multiscale patches from the voxel backbone maps via tri-linear interpolation. Once sampled, the derived volumes concatenate onto the points' feature matrices directly. Final integration involves mapped MLPs that inject this density continuity onto the points' sparse representation smoothly.

**3. Recycle Mechanism:**
Inspired by iterative refinement strategies seen in AlphaFold architectures, our model natively encompasses a finite recycle routing procedure (averaging 1 to 3 rounds). In between successive passes, previous hidden states emitted from the voxel backbone (`voxel_recycle_out`) and point backbone (`point_recycle_in`) bypass standard data drops, redirecting backward as disconnected, detached gradient residual inputs. Notably, processing components such as the `embed head` and `atom head` reside entirely outside the cycle loop—ensuring that cross-modal representation tuning scales robustly without inflating excessive memory per step.
