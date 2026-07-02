# Research Roadmap — DensitySwinSaliency

Current state: NSS 1.434 ± 0.007 (mean over 4 seeds), +14.7% over ACL-Net. All peer-review edits complete in `report/main.tex`. This document captures the extended reviewer Q&A — specifically what they would have built differently — and derives a phased work plan from it.

---

## Reviewer Panel: Extended Q&A

Three domain specialists were consulted. Questions asked of each:
1. How would you have designed the model differently?
2. How would you improve it from here?
3. What other relevant datasets should be used?
4. What ideas could yield a substantially better model?

---

### R1 — Video Saliency (CVPR/ECCV reviewer)

**How would you have designed it?**

The fundamental bet — density conditioning at the bottleneck — is the right instinct, but I'd have distributed the conditioning signal across decoder stages, not applied it once at the bottleneck. The decoder's shallow stages (stride 4, stride 8) are where spatial saliency decisions are actually made; the bottleneck at stride 32 is too coarse to steer fine-grained fixation patterns. I'd have applied FiLM at each of the 4 decoder scales with shared density embeddings, so density information propagates through the full resolution hierarchy. One FiLM application gives one shot to influence the output; four gives continuous steering.

I also would have decoupled the density estimator: train a dedicated crowd counting head on a large-scale counting dataset (CSRNet backbone, trained on ShanghaiTech or UCF-QNRF) and freeze it before plugging into saliency training. An auxiliary head trained jointly on 3 classes in a 5,356-clip dataset is weak supervision — the density representation is almost certainly not as clean as one trained on explicit crowd density regression with thousands more examples.

**How would you improve it from here?**

In rough priority order:

1. Multi-scale FiLM: apply the (γ, β) pairs at decoder strides 32, 16, 8, 4. Shared projection from the same density embedding. Expected gain: +0.02–0.04 NSS. One architectural change, one training run.
2. Temporal consistency loss: add `L_temporal = ||sal(t) - warp(sal(t+1), flow)||_1` where flow comes from RAFT. Encourages smooth saliency propagation across frames without flickering. Should help especially in DF/DC categories where crowds move coherently.
3. Replace argmax density conditioning with continuous softmax probabilities (no extra parameters, handles ambiguous SP→DF clips). Already on your roadmap; do this before multi-scale FiLM so you can attribute gains cleanly.
4. Replace Kinetics-pretrained VideoSwin with a crowd-specific pretrained encoder. VideoMAE pretrained on crowd surveillance footage, or a model fine-tuned on UCF-QNRF density maps, would bring a much richer representation of crowd dynamics.

**What other relevant datasets?**

- **DHF1K** (300 training / 100 test dynamic scenes, 600 fixators): the standard video saliency benchmark. Zero-shot eval here would validate generalisation; fine-tuning would likely improve CrowdFix results via richer motion priors.
- **LEDOV** (Large-scale Eye-tracking Dataset on Videos, 538 videos, 32 fixators): diverse video content with consistent fixation protocol. Good pretraining source.
- **UCF-QNRF** (1,535 images, 1.25M annotated heads): no fixations, but richest available crowd density annotation. Use to pretrain/fine-tune the density estimation head.
- **ShanghaiTech** (Parts A+B, 1,198 images): similar use case — density head pretraining.
- **GazeCom** (18 dynamic scenes, 54 observers, gaze + head pose): low-level but has gaze velocity which would support a temporal saliency consistency metric.

**What would yield a substantially better model?**

Foundation model encoder. The CrowdFix training set is 3,189 clips — large for saliency, small for representation learning. A VideoMAE-Large encoder pretrained on crowd surveillance footage (CCTV datasets, UCF-Crime, ShanghaiTech video extension) and then fine-tuned on CrowdFix would bring 10–100× more crowd-specific priors than Kinetics. This alone could close the remaining gap to a human upper bound, independent of any architectural change to the conditioning pathway. Estimated effort: 2–3 LUMI jobs, 1 week.

---

### R2 — Cognitive Science / Eye-tracking

**How would you have designed it?**

The model treats density as a scene descriptor. It isn't. Density is a scene *property* that mediates which fixation *strategy* an observer will use. A more principled design would have modelled the strategy directly, not the scene property. In SP scenes observers track individuals; in DF they track coherent flow; in DC they scan for local contrast and proto-objects. These are distinct oculomotor programs, not a scalar density gradient.

My design would have: (1) a perceptual-primitive stream computing biological low-level saliency (Itti-Koch proto-objects, motion energy, temporal contrast) and (2) a social-attention stream detecting gaze direction, body orientation, and interaction geometry of individuals in the frame. Crowd density then acts as a gating variable between these two streams, not as a direct conditioning signal. In sparse crowds the social stream dominates; in dense congested crowds it degrades (too many individuals, gaze directions cancel) and the proto-object stream takes over. This matches known psychophysics.

**How would you improve it from here?**

1. Within-category analysis: partition the SP category's test clips by group size (1 person, 2–5 persons, 5–15 persons) and plot NSS as a function of group size. I predict a U-shaped curve — individual tracking is easy, small-group social dynamics are hard, large group reverts to flow tracking. This is a zero-cost analysis on your existing results and tells you exactly where the model fails.
2. Add a gaze-direction auxiliary head: for any frame where individuals are detectable (SP/low-DF), extract face orientation from a lightweight OpenCV/MTCNN head and add a gaze-direction prior heatmap as a second conditioning input. This is the social attention signal that R1's "coherent flow" doesn't capture.
3. Social attention mode conditioning: extend from 3 density classes to a 2D label (density × mode: tracking / scanning / flow-following). Requires behavioural annotation of existing CrowdFix eye-tracking data — reviewers coding ~200 clips, ~3 weeks — but this is the single change most likely to yield publishable psychophysical results alongside the saliency improvement.

**What other relevant datasets?**

- **SocEye** (social attention in naturalistic scenes, 1,008 images, gaze from 22 observers): directly relevant for the social attention stream.
- **CAT2000** (4,000 images, 20 categories including crowd/urban): contains crowd scenes, 24 fixators, high-quality saliency maps. Good for pretraining the decoder.
- **SALICON** (10,000 images from MS-COCO, 68 fixators via mouse-contingent simulation): large-scale, useful for decoder initialisation.
- **MIT300** and **MIT1003**: standard image saliency benchmarks — not crowd-specific but useful for verifying the decoder doesn't overfit CrowdFix statistics.
- **EyeTrackUAV** (drone footage of pedestrians and crowds): a less-known dataset that bridges crowd density and motion saliency from an elevated viewpoint; very different centre bias from CrowdFix.

**What would yield a substantially better model?**

A proper behavioural experiment. Crowd scenes activate social attention circuits that general saliency models are not trained to predict. If you can get 20–30 observers to free-view a subset of CrowdFix clips (even 200 clips) while coding their fixation strategy (tracking a specific person vs. scanning the crowd), you'd have ground truth for the second conditioning axis. A model trained on this 2D label space would be the first to explicitly separate low-level motion saliency from high-level social attention in crowds — that is a Nature Comms paper, not a CVPR paper.

---

### R3 — ML/Reproducibility

**How would you have designed it?**

I would have started with a much simpler model and built up from ablation evidence rather than proposing the full DensitySwin architecture upfront. Specifically:

Baseline 0: VideoSwin encoder + 2D decoder, no density conditioning, trained with fixed seed. Get a clean number.  
Baseline 1: Same, + 1D centre-bias prior added as a constant spatial prior. How much of your gain is just centre bias?  
Baseline 2: Same as 0, + one-hot density bias (additive, no embedding). How much is the conditioning signal alone?  
Model: Same as 2, + FiLM embedding. Does the 64-dim embedding geometry add value over the one-hot bias?

You ran baselines 0 and 2 (eventually), but baseline 1 (centre bias) was never evaluated. Without it, a reviewer cannot distinguish "the model learned to predict crowd fixations" from "the model learned a dataset-specific spatial prior." Given that VideoSwin was pretrained on Kinetics which has a strong centre bias, this is a real confound.

**How would you improve it from here?**

1. **Debiased NSS** (most urgent): subtract the mean fixation heatmap from all predictions before computing NSS. Report both biased and debiased numbers. One eval job, no retraining. This closes the centre-bias question permanently.
2. **Zero-shot eval on a second crowd dataset**: take the existing checkpoint and run inference on DHF1K or EyeTrackUAV without any fine-tuning. If NSS is above random on held-out sequences, the representation generalises; if it collapses, you have a dataset-specific model and should say so. No training required.
3. **Test-time ensembling**: at inference, run 4 seed checkpoints and average predictions. Trivial to implement, likely +0.003–0.005 NSS on Table 1. Worth the one sentence in §3.
4. **Pretrained density estimator as a frozen feature**: replace the jointly-trained 3-class head with CSRNet or BayesCrowd features (pretrained on ShanghaiTech Part A, frozen). This changes the density signal from "3-class argmax with weak CrowdFix supervision" to "continuous density map from a dedicated counting network." Expected gain: +0.01–0.03 NSS, one training run, addresses R1's concern about auxiliary head quality.
5. **Cross-dataset fine-tuning**: pretrain on DHF1K (larger, diverse dynamic scenes), fine-tune on CrowdFix. This has been shown to improve video saliency by ~5–8% NSS on target datasets. Effort: 2 LUMI jobs.

**What other relevant datasets?**

- **UCF-QNRF** (1.25M head annotations, 1,535 images): for pretraining the density estimation head.
- **ShanghaiTech Part A+B**: same use case, more commonly used for CSRNet/BayesCrowd pretraining.
- **DHF1K** (300 training video clips, diverse dynamic scenes, 17 fixators): mainstream video saliency benchmark; pretraining source and zero-shot generalisation test.
- **Hollywood-2 Saliency**: action recognition clips with gaze data; good domain coverage for fine-grained motion.
- **AVS1K** (1,000 audiovisual scenes with gaze): if you ever extend to audio-guided attention in crowd videos (crowd noise and saliency are correlated).

**What would yield a substantially better model?**

Two things in combination: (1) a pretrained crowd density encoder (CSRNet or BayesCrowd features, not a 3-class head), and (2) a multi-dataset pretraining curriculum — DHF1K (general video saliency) → CrowdFix (crowd-specific fine-tuning). The density encoder provides a continuous representation of crowd structure; the pretraining curriculum ensures the saliency decoder has seen enough motion diversity to separate crowd motion from camera motion. Together, estimated +0.05–0.08 NSS. These are two independent changes you can run sequentially on LUMI with no architectural redesign beyond swapping the density head.

---

## Roadmap

Derived from the above. Ordered by NSS return per effort.

---

### Phase 0 — Writing fixes (this week, no LUMI, 1 day total)

All items are edits to `report/main.tex`.

| # | Action | Section(s) | Effort |
|---|--------|-----------|--------|
| 0a | Reframe contribution bullet: "bottleneck density conditioning" not "FiLM mechanism"; one-hot ablation (89% of gain) justifies this | §1 contributions, abstract | 1 h |
| 0b | Add explicit 1.9σ qualification sentence: FiLM vs one-hot is indicative, geometric γ evidence is independent support | §5.4 | 30 min |
| 0c | Add centre-bias limitation paragraph citing VideoSwin Kinetics pretraining; forward-ref to debiased NSS experiment | §6 Limitations | 30 min |
| 0d | Add within-category SP group-size note as a zero-cost observation (R2 observation) | §5.3 | 30 min |

---

### Phase 1 — Post-hoc analysis, no retraining (~3 days, 1–2 LUMI eval jobs)

#### 1a: Debiased NSS
Subtract the mean GT saliency map (averaged over all 3,189 training clips) from each model prediction before metric computation. Re-run `evaluate.py` with this normalisation.

```
# Compute mean GT saliency over training split
python scripts/compute_mean_gt.py --split train --out results/mean_gt_saliency.npy

# Add --debias flag to evaluate.py
python evaluate.py --checkpoint checkpoints/best.pth \
    --density-mode oracle \
    --debias results/mean_gt_saliency.npy
```

Expected result: debiased NSS drops ~0.03–0.05 (centre-bias contribution), still clearly above ACL-Net 1.250. Add as "DensitySwinSaliency (debiased)" row in Table 1. If the drop is larger than 0.08, add a Limitations paragraph that quantifies it honestly — this is the publishable finding either way.

#### 1b: Centre-bias upper bound
Evaluate a trivial "predict the mean GT saliency map" baseline. This is the ceiling for centre-bias exploitation.

```
python evaluate.py --baseline centre-bias \
    --centre-bias-map results/mean_gt_saliency.npy
```

Add as "Centre bias (upper bound)" row in Table 1 below all learned models.

#### 1c: Within-category SP analysis (zero-cost)
Partition SP test clips by approximate group size (1 person, 2–5, 5+) using the crowd head count from the density estimator or a simple frame-level head detector. Plot NSS by group-size bin. Check whether the U-curve R2 predicted appears. Writeup in §5.3 if interesting; appendix if minor.

---

### Phase 2 — LUMI experiments, current architecture (~1 week, 2–3 jobs)

#### 2a: Continuous density conditioning
Replace `argmax(density_logits)` with `softmax(density_logits)` as the FiLM input. The model already computes the softmax internally; this is piping it back rather than discarding it.

- `models/density_swin_saliency.py`: change `density_emb(density_class_idx)` to `(softmax_probs @ density_emb.weight)` — a weighted sum of the three embedding vectors
- `train.py`: pass `F.softmax(logits, dim=-1).detach()` as the conditioning input during the auxiliary-head forward pass
- One seed train (~40 min on LUMI) + eval

Expected: +0.01–0.02 NSS. Add row to Table 2.

#### 2b: Multi-scale FiLM
Apply the density conditioning at all 4 decoder scales (strides 32, 16, 8, 4) with shared projection weights. Each decoder stage gets its own affine recalibration steered by the same density embedding.

- Modify `models/density_swin_saliency.py` decoder stages to accept `(gamma, beta)` from `film.proj(density_emb(d))`
- Same 64-dim embedding, same FiLM module, 4× application points
- One seed train + eval

Expected: +0.02–0.04 NSS on top of 2a. This is the highest-expected-return architectural change achievable without new data or a new encoder.

#### 2c: Pretrained density head (CSRNet features, frozen)
Replace the jointly-trained 3-class auxiliary head with frozen CSRNet features pretrained on ShanghaiTech Part A. This gives continuous density maps rather than a 3-class categorical label, and brings in ~1,200 strongly supervised crowd images worth of density knowledge.

- Download CSRNet checkpoint (publicly available)
- Extract density map for each frame; pool spatially to a 64-dim descriptor
- Feed this descriptor into FiLM in place of the learned `density_emb` lookup
- Requires one training run (density head is frozen; only saliency decoder trains)

Expected: +0.01–0.03 NSS, and resolves R1/R3's concern about auxiliary head quality. If this outperforms the learned 3-class head, it also strengthens the "conditioning signal, not mechanism" narrative.

---

### Phase 3 — Cross-dataset pretraining (~2 weeks, 3–4 LUMI jobs)

#### 3a: DHF1K pretraining
Pretrain on DHF1K (300 training clips, 17 fixators, diverse dynamic scenes) then fine-tune on CrowdFix. VideoSwin encoder stays frozen; only the decoder and FiLM head are trained.

- Download DHF1K (publicly available, ~12 GB)
- Adapt `crowdfix_dataset.py` to a `dhf1k_dataset.py` with the same interface
- Pretrain: `train.py --dataset dhf1k --epochs 30`
- Fine-tune: `train.py --dataset crowdfix --pretrained checkpoints/dhf1k_best.pth`

Expected: +0.03–0.06 NSS based on published cross-dataset transfer results in video saliency literature. This is the single highest-expected-return experiment that doesn't require new data collection.

#### 3b: Zero-shot eval on DHF1K / EyeTrackUAV
After 3a, evaluate the CrowdFix-trained checkpoint (without DHF1K fine-tuning) on DHF1K test splits.

```
python evaluate.py --checkpoint checkpoints/best.pth \
    --dataset dhf1k --split test
```

Report as "zero-shot generalisation" in §5 or a new §6. If NSS is above random (≥0.6), the representation generalises; write this as a positive result. If it collapses, report it honestly in Limitations — a dataset-specific model is still a valid contribution given CrowdFix's unique annotation scheme.

---

### Phase 4 — Next paper (months, new data or major retraining)

These items are out of scope for the current submission. File here for the follow-up paper.

#### 4a: Social attention stream (R2)
Two-stream architecture: (1) low-level saliency stream (current DensitySwin), (2) social attention stream (face detector → gaze orientation heatmap → learned social prior). Density gates between the two streams.

Requires: MTCNN or a lightweight face/body orientation detector. Can reuse CrowdFix frames. No new eye-tracking needed for the architecture, but new annotation is needed for the 2D (density × mode) conditioning below.

#### 4b: 2D conditioning (density × fixation strategy) (R2)
Extend labels from 3 density classes to a 2D space: density (SP/DF/DC) × fixation strategy (individual-tracking / group-scanning / flow-following). Requires behavioural annotation of ~200–400 CrowdFix clips by 2–3 trained annotators. Approximately 3 weeks of annotation work.

This is the experiment that closes the gap between "scene label" and "fixation strategy" — the psychophysically correct framing of the problem.

#### 4c: Foundation model encoder (R1)
Replace VideoSwin-S (Kinetics pretrained) with VideoMAE-Large pretrained on crowd surveillance footage (UCF-Crime + ShanghaiTech video extension + custom CCTV clips). Requires pretraining compute beyond LUMI small-g allocation or a compute grant.

#### 4d: Test-time ensembling (R3, quick win)
Average predictions across all 4 seed checkpoints at inference. No retraining. +0.003–0.005 NSS. Add to paper as a one-line "ensemble" row in Table 1 whenever implementing any of phases 1–3.

---

## Implementation order

```
Week 1: Phase 0 (writing) + Phase 1a–b (debiased NSS, centre-bias baseline)
Week 2: Phase 2a (continuous conditioning) + Phase 1c (SP group-size analysis)
Week 3: Phase 2b (multi-scale FiLM) 
Week 4: Phase 2c (CSRNet density head) + recompile paper with all new rows
Week 5+: Phase 3 (DHF1K pretraining) if submitting to a venue that expects cross-dataset results
Future: Phase 4 items → new paper
```

---

## Open questions before starting Phase 2

- [ ] Is CSRNet checkpoint accessible in the LUMI container environment or does it need to be added to the sqsh?
- [ ] Is DHF1K download link stable (original hosting was at University of Amsterdam)?
- [ ] For 2b multi-scale FiLM: confirm decoder stage interfaces in `models/density_swin_saliency.py` accept feature-level hooks cleanly before designing the FiLM injection points.
