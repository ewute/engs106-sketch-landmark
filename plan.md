# The Plan

## Raw Notes

We need a data source, to do something to the data, and then some analysis (quantitative or qualitative). 

I plan for us to use the CUHK Face Sketch Database (CUFS). There are a total of 606 faces. 188 faces from the Chinese University of Hong Kong (CUHK) student database, 123 faces from the AR database, and 295 faces from the XM2VTS database so we need to split them intelligently. There's also 1194 sketches along these faces. I think a really cool metric to test our result would be in the testing phase, we can pull up 5 or so faces and ask it who it matches to. Some further literature could be generating these faces using some image generation model and testing if it works as well.

So ideally, we have a cnn that's trained to identify facial features, and we can use transfer learning. I'm not 100% how we should set up the training and learning though.

### Additional Notes

- Since there are 1194 sketches for 606 faces, some faces have multiple sketches (~2 per face on average). This is useful for augmentation / showing the model different artistic styles.
- The three sub-databases have different lighting, pose, and image quality characteristics — this is both a challenge and an opportunity (domain generalization).
- Transfer learning with a pretrained face recognition backbone (e.g., VGGFace, ArcFace, FaceNet) is a strong starting point since those models already understand facial features.
- A **Siamese network** architecture is a natural fit here: pass in a sketch and a photo, learn whether they match. At inference time, rank candidates by similarity.

---

## Structured Outline (TODO — fill in details)

### 1. Problem Statement

### 2. Dataset
- **Source:** CUHK Face Sketch Database (CUFS)
- **Size:** 606 identities, 1194 sketches
- **Sub-databases:**
  - CUHK Student (188 faces)
  - AR (123 faces)
  - XM2VTS (295 faces)
- **Preprocessing:**

- **Split strategy:** Random identity-based split (~70/15/15 train/val/test). Ensure all sketches for a given identity stay in the same split.


### 3. Model Architecture
- **Backbone:** VGGFace2 / ArcFace (pretrained on faces — should transfer well to sketch domain)
- **Architecture pattern:** Siamese network — learn a shared embedding space for sketches and photos, compare via cosine similarity
- **Loss function:** Contrastive loss (pull matching sketch-photo pairs close, push non-matching pairs apart)

- **Frozen vs. fine-tuned layers:**


### 4. Training
- **Optimizer & learning rate:**

- **Batch size & epochs:**
- **Pair/triplet mining strategy:**
- **Hardware:** Google Colab A100 GPU

### 5. Evaluation Metrics
- **Rank-1 accuracy:** Given a sketch, is the correct photo the top-1 match?
- **Rank-5 accuracy:** Is the correct photo in the top-5 matches?
- **Retrieval visualization:** Show a sketch alongside its top-5 predicted photos.
- **Per-database breakdown:** Performance on CUHK vs AR vs XM2VTS subsets.

### 6. Experiments & Ablations
- [ ] Baseline: pretrained CNN features + nearest neighbor (no fine-tuning)
- [ ] Fine-tuned Siamese network
- [ ] Effect of different backbones
- [ ] Effect of data augmentation
- [ ] Cross-database generalization (train on one subset, test on another)
- [ ] (Stretch) Generate synthetic sketches and test matching


### 7. References
- CUFS Database: http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html
- Wang & Tang, "Face Photo-Sketch Synthesis and Recognition" (TPAMI 2009)
- 