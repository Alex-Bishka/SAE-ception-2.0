# SAE-ception 2.0: Remaining Work Checklist

> **Instructions**: Check items by changing `[ ]` to `[x]` as you complete them.

---

## 🔴 HIGH PRIORITY (Required for Valid Experiment)

### Missing Baselines

- [x] **Run B: L1 Penalty Baseline**
  - Add CPT run with `L_next_token + λ||x||₁` (no SAE)
  - Purpose: Prove SAE-ception provides unique benefits beyond simple regularization
  - Deliverable: New script or flag in `train_cpt.py`

### Missing Evaluation Scripts

- [x] **Unified Perplexity Comparison Script**
  - Compare: Baseline → Control (no aux) → L1 Penalty → SAE-ception
  - Output: Single table with PPL and % degradation
  - Tax Check: Verify degradation <2%

- [ ] **Structure Check Analysis Script**
  - Compare SAEs trained on control vs SAE-ception models
  - Metrics to compare: L0 sparsity, dead features %, convergence speed
  - Hypothesis: SAE on SAE-ception model should be "lazier"

### Workflow Validation

- [ ] **k_sharp Selection Workflow**
  - Enforce: Run intervention baseline BEFORE CPT
  - Document: Clear process to pick k_sharp from degradation curve
  - Current gap: Smoke tests don't enforce this ordering

---

## 🟡 MEDIUM PRIORITY (Strengthens Claims)

### Experimental Rigor

- [ ] **Multiple Seeds / Variance Reporting**
  - Run same experiment with 3-5 different seeds
  - Report: mean ± std for all metrics
  - Addresses reviewer criticism from original paper

- [ ] **λ (aux_weight) Sweep or Justification**
  - Research plan says: "Start with λ=1.0, anneal if perplexity explodes"
  - Current default: 0.01
  - Either: Justify current value OR add sweep

- [ ] **Random Target Ablation**
  - CPT with random unit vectors as targets (same k, random directions)
  - Should NOT improve structure
  - Proves sharpening is necessary (not just regularization effect)

### Downstream Evaluation

- [ ] **Ghost Check: BLIMP or Logic Tasks**
  - Evaluate on Linguistic Minimal Pairs (BLIMP)
  - Or simple syntactic agreement tasks
  - Hypothesis: Clean representations → better syntactic performance

---

## 🟢 LOWER PRIORITY (Paper Polish)

### Scale Testing

- [ ] **Scale to Pythia-410M**
  - Research plan: 70M (dev) → 410M (validation) → 1.4B (scale)
  - Verify method works beyond toy scale

- [ ] **Scale to Pythia-1.4B**
  - Final scale test
  - May require gradient checkpointing / memory optimization

- [ ] **Proper Sample Sizes**
  - Current smoke test: 5k SAE samples, 1k CPT samples
  - Paper-ready: 100k-500k tokens
  - Add `--full` flag or separate config

### Additional Metrics

- [ ] **Feature Absorption Score (SAEBench)**
  - `saebench.py` exists but unclear if integrated with LLM pipeline
  - Measures if features stop firing when expected

- [ ] **Per-Token Analysis**
  - Track which tokens are most affected by sharpening
  - Debugging insight for understanding method behavior

### Infrastructure Improvements

- [ ] **W&B Integration for CPT**
  - `train_cpt.py` lacks W&B logging
  - Needed for tracking experiments at scale

- [ ] **Track Aux/LM Loss Ratio Per Step**
  - Diagnostic: Is aux loss dominating? Negligible?
  - Helps tune λ

- [ ] **Unified Results Summary Script**
  - Current: Multiple `.diagnostics.json` files
  - Need: Script that aggregates and prints comparison table

---

## 🔵 CODE FIXES / IMPROVEMENTS

### Verification Needed

- [ ] **Verify MSE vs Cosine Loss Intentional**
  - `train_cpt.py` (LLM): Uses MSE ✅
  - `auxiliary.py` (classification): Uses cosine distance
  - Confirm this difference is intentional for different tasks

- [ ] **Checkpoint Naming Consistency**
  - Mix of "cycle" (vision paper) and "cpt" terminology
  - Consider standardizing for clarity

### New Smoke Tests to Add

- [ ] **Smoke Test: L1 Baseline**
  - Step 2b: CPT with L1 penalty (no SAE)
  - Step 5b: Train SAE on L1 model
  - Compare to control and SAE-ception

- [ ] **Smoke Test: Full Evaluation Pipeline**
  - After all CPT runs, evaluate all models
  - Single comparison table
  - Fail condition if SAE-ception degrades >5%

- [ ] **Smoke Test: Random Target Ablation**
  - CPT with random targets
  - Verify it does NOT improve structure

---

## 📊 PROGRESS TRACKER

| Category | Done | Total | Progress |
|----------|------|-------|----------|
| High Priority | 0 | 4 | ⬜⬜⬜⬜ |
| Medium Priority | 0 | 4 | ⬜⬜⬜⬜ |
| Lower Priority | 0 | 8 | ⬜⬜⬜⬜⬜⬜⬜⬜ |
| Code Fixes | 0 | 5 | ⬜⬜⬜⬜⬜ |
| **TOTAL** | **0** | **21** | **0%** |

---

## 📝 NOTES

### Decisions Made
<!-- Record decisions here as you make them -->


### Blockers
<!-- Record any blockers here -->


### Ideas for Later
<!-- Park ideas that aren't urgent -->


---

*Last updated: [DATE]*