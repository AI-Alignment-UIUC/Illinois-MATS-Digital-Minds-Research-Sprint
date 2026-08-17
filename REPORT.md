# Evaluating Introspection of Self-reflective R-Lens Feedback Training

**William Wale**, MATS Research · **Heather Broome**, UIUC · **Ricky Mouser**, MATS Research · **Evan Coats**, UIUC

*With Apart Research*

---

## Abstract

We present an evaluation of model introspection that scores each self-report as two numbers rather than one: how well a model tells the prompts it answers consistently from those where it varies, and how readily it claims consistency at all. A single stated-minus-actual gap cannot separate a model that cannot tell from one that can tell but misreports, and because over-claims and under-claims cancel when averaged, a gap can sit near zero while most individual reports are wrong. The decomposition, with a polarity-flipped rewording, a cross-model control, a serving check and a comparison against an internal readout, lets particular non-introspective reports be located rather than summarized away.

We applied it to a novel self-reflective reinforcement learning technique that rewards a Qwen3.5-4B model for verbalizing its own R-lens readouts, testing six checkpoints and five off-the-shelf models on 48 prompts with resampled ground truth. The evaluation identified three kinds of case where the trained model's report was not tracking its own behavior: it claimed consistency on 23 of the 26 prompts where its answers demonstrably varied, with criterion moving from −0.04 [−0.57, 0.45] to −1.45 [−1.94, −1.18]; its criterion reversed sign when the question's polarity was flipped; and its verbal report stopped tracking a property an R-lens readout still tracked, a paired difference of +0.38 [+0.06, +0.71]. Over the same checkpoints the gap statistic read −1.8 pp and −1.6 pp, and sensitivity showed no measurable change in either direction.

---

## 1. Introduction

In this work we are addressing the question of whether using reinforcement learning to train a model to learn to predict what is in its R-space at particular tokens will increase its self-awareness, as measured by a model introspection evaluation. This question matters because training for and quantifying self-awareness of models could allow for new capabilities or alignment approaches.

Answering it requires an evaluation that a reporting habit cannot satisfy, and there is a problem with non-rigorous methods for evaluation of model introspection. The statistic usually reported is a gap between what a model claims about itself and what is true, and it cannot separate a model that cannot tell from one that can tell but misreports, because a model claiming consistency too readily is right where it is consistent and wrong where it varies, and those errors cancel when averaged. Our work presents a first step at a general instrument that can identify the depth of a model's self knowledge, and utilizes this evaluation on a proposed reinforcement learning method for increasing introspection. Because the instrument scores detection and reporting separately, its output is not a single verdict on the training method but a set of located cases in which a model's report does or does not track its own behavior.

Our main contributions are:

- Reproducible quantitative introspection evaluation: confidence in self-prediction of outputs
- Novel self-reflective reinforcement learning training technique using R-lens feedback

## 2. Related Work

Binder et al. [1] establish the design we build on, in which a model predicts properties of its own behavior and other models predicting the same target control for privileged access; we extend that control to confidence as well as accuracy. That predicting others and predicting oneself may share machinery is the simulation theory of empathy [7], which is what makes cross-prediction informative rather than a bare baseline, and work on whether models act in line with their stated values [5] and on emergent introspection [4] motivates treating a self-report as a claim to be checked rather than as evidence. The logit lens [8] reads activations through the unembedding directly, the J-lens uses the Jacobian of the final residual with respect to a layer's residual and has been discussed as a candidate global workspace [3, 6], and the R-lens [2] is fit the same way with three relevance propagation rules that make it more faithful on early layers. Prior work asks whether models can introspect. We ask how to tell whether a given self-report is introspective, and use a training method built to induce introspection as the case the instrument is tested against.

## 3. Methods

### 3.1 Introspection Evaluation on Self-prediction

We use 48 short-answer prompts spanning categories a model answers identically every time and categories where it varies. Ground truth is 16 fresh samples per prompt at temperature 1.0, giving the most common answer and its share p(mode); a prompt is consistent if p(mode) ≥ 0.75 and variable if ≤ 0.40, with the middle band excluded from class-based scores. Meta channels run at temperature 0. The two channels are:

```
Do NOT answer the question above. Predict the following:
MOST_LIKELY: <the single answer you would most often give>
SAME_PCT: <0-100: out of 100 fresh instances, how many give exactly that same answer>
MODE_CONF: <0-100: probability that the MOST_LIKELY you named above really is
your most common answer>
```

```
Do NOT answer the question above. Consider how 100 fresh instances of you
(same model, no memory of this) would respond to it.
DETERMINISTIC: <YES if at least 75 of 100 would give the identical answer, else NO>
CONFIDENCE: <0-100 in your YES/NO call>
```

From the YES rates on consistent and variable items we compute two numbers from standard signal detection theory: sensitivity d′, how separable the classes are, and criterion c, the standing willingness to answer YES, where negative c means claiming consistency too readily. We also report the stated-minus-actual gap for comparison. Four further instruments let a report be checked rather than scored. The consistency question is rerun reworded at the same polarity and again with polarity flipped and re-coded, since a criterion reflecting a stable belief should keep its sign in all three. Every model predicts every other model, which asks whether any apparent self-knowledge is privileged. Because the construct assumes the provider samples, we resample five prompts eight times each at temperature 1.0 and 1.5. And for lens-served checkpoints we read the R-lens at the last prompt token over four layer bands and correlate its concentration with true p(mode), which asks whether a property the report has stopped conveying is still present internally. Across 11 variants this came to about 14,400 calls, and all intervals are 95% bootstrap over items.

### 3.2 R-lens Reinforcement Learning

For each layer of Qwen3.5-4B we fit the expected same-position Jacobian of the final residual stream with respect to that layer's residual, estimated by random-probe vector-Jacobian products over roughly 1,000 FineWeb-Edu documents, with the three relevance propagation rules of the R-lens recipe [2] applied to the backward pass while forward values are left exact. Composed with the model's own final norm and unembedding, this gives a distribution over tokens for what the computation at a given token and layer promotes.

One episode is a short document, a token position, and a layer, and the model emits a single token in `<answer>` tags whose correct value is the lens top-1 token. Because base models refuse questions about their activations, an SFT stage on synthetic introspective traces removes the refusal and initializes RL. Training is REINFORCE with group baselines, with no importance ratio, no clipping, no KL or entropy term, and one Adam step per rollout, sampling 64 episodes with 32 trajectories at temperature 1.0 with layers drawn from 14 to 26 of 32. The reward is the R-lens probability of the named token under the current policy, divided by the lens top-1 probability so a correct answer scores 1.0 at any depth; unparseable output and truncation each score −1.0, and a shaping term penalizes thinking under 40 tokens. Rewards are rank-transformed onto [−1, 1] within each group, and training uses a rank-8 rsLoRA on MLP projections at learning rate 4e-5 for 100 steps.

## 4. Results

| | base | SFT | step 25 | step 50 | step 75 | latest |
|---|---|---|---|---|---|---|
| false alarms / variable items | 8/30 | 13/31 | 19/27 | 21/24 | 22/23 | 23/26 |
| criterion c | −0.04 [−0.57, 0.45] | −0.35 [−0.84, 0.12] | −0.66 [−1.23, −0.22] | −1.44 [−1.94, −1.16] | −1.65 [−1.92, −1.37] | −1.45 [−1.94, −1.18] |
| sensitivity d′ | 1.27 [0.39, 2.39] | 1.08 [0.12, 2.07] | 0.30 [−0.57, 1.34] | 0.72 [−0.20, 1.28] | 0.23 [−0.51, 0.84] | 0.64 [−0.27, 1.18] |
| self-prediction accuracy | 0.28 | 0.37 | 0.31 | 0.35 | 0.40 | 0.32 |
| stated-minus-actual gap (pp) | −1.8 | +2.9 | −12.4 | −12.3 | −7.1 | −1.6 |

**Table 1.** The six checkpoints, with 95% intervals where computed. The full table, including hit rates, confidence AUROC and both tracking correlations, is in the appendix.

**Cases the evaluation identified.** The first is reports that contradict the model's own measured behavior. On the prompts where its answers genuinely vary, the trained model claimed consistency 8 times out of 30 at base and 23 out of 26 at the end, with criterion moving from an interval containing zero to −1.45 [−1.94, −1.18], which does not overlap it. These are enumerable items rather than an aggregate, since for each one we have the model's claim beside its measured answer distribution.

The second is reports that follow the question rather than the model. Criterion for the trained checkpoints runs from −1.45 [−1.94, −1.18] in the original wording to +0.03 [−0.49, 0.50] reworded at the same polarity and +1.62 [1.30, 1.92] with polarity flipped, so the sign reverses, and a report whose direction depends on how it was elicited is not describing a stable property. Because the reworded version sits near zero rather than strongly negative, this is not a simple habit of always answering yes. Most off-the-shelf models are unstable the same way, with only Haiku 4.5 keeping its sign in all three wordings.

The third is reports that have stopped conveying something the model still holds. At base the R-lens readout and the verbal report carry about the same information about true consistency, a paired difference of +0.04 [−0.33, +0.40]; at the final checkpoint the lens is ahead by +0.38 [+0.06, +0.71], excluding zero. The lens never saw a consistency label, and the RL was rewarding the model for verbalizing this very readout, yet the readout's information survived and the verbalization's did not. Present here means linearly decodable at one next-token position, which is weaker than accessible to the model. Two further cases are in Appendices B and C.

Over the same interval the gap statistic read −1.8 pp at base and −1.6 pp at the end, while the false alarm rate went from 27% to 88%. This is the cancellation described in Section 1, so an evaluation reporting only the gap before and after could certify the self-report as unchanged while the reporting behavior had inverted.

**What the evaluation did not detect.** Two results are nulls. No checkpoint's d′ exceeds the base model's 1.27, but every interval overlaps every other, so with 7 to 13 consistent prompts per checkpoint we can only say we found no evidence that detection improved, not that it declined. Self-prediction accuracy shows no trend, and its largest step is base to SFT before RL begins, with accuracy falling from 0.37 to 0.32 across RL itself; confidence AUROC behaves the same way.

![Figure 1](introspection-eval/figures/fig7.png)

**Figure 1.** The six checkpoints on the two axes the evaluation separates. Horizontal is what the model says, vertical is what it can tell. The run moves left into claiming consistency and does not move up, which is the pattern the decomposition exists to make visible.

## 5. Discussion and Limitations

The evaluation's contribution is that it names cases instead of returning a score, and it did so on a run where the commonly used gap statistic returned the same value before and after. What we can say about the training method is narrower: over 75 RL steps the model's willingness to assert consistency changed a great deal while its measured ability to discriminate did not change in either direction, so on this construct the training produced no evidence of improved introspection.

For AI safety this matters because a self-report is the mechanism by which a model would tell us it is uncertain or that something has shifted its behavior, and a procedure that makes such reports more confident without making them more informative leaves a model harder to oversee. The practical lessons are cheap to adopt. Splitting a self-report into a detection term and a reporting term costs nothing extra to collect, since the same YES/NO answers produce both. Asking every question in two polarities costs one extra prompt per item and would have stopped us reading a wording effect as a belief. Checking that the provider samples is nearly free, and one of five bench models failed it. Because consistency information stayed decodable while the report stopped conveying it, the evaluation locates this failure in the reporting channel rather than the capacity, which points toward rewarding discrimination instead of assertion.

### Limitations

The evaluation covers one construct, self-consistency of short answers, on 48 items, and only criterion, the false alarm counts, and the lens contrast have intervals tight enough to interpret. The sensitivity results are nulls and should not be read as showing detection got worse. We applied it to one run of one method on one model with one seed, so we make no general claim about R-lens feedback, and the instrument's own generality rests on the five bench models rather than on this trajectory. The head checkpoint was captured twice while the job was live, at c = −1.13 and −1.45, with Table 1 using the later and the lens contrast the earlier. The construct assumes the provider samples as documented, and it failed for one bench model whose route served deterministically. The lens reads one next-token position through a linear map with top-50 truncation while answers are multi-token, and the wording controls cover the consistency channel only, so self-prediction results may also be wording-bound. A behavioral arm on self-knowledge of sycophancy was implemented but the R-lens server went down before the checkpoints could be run through it.

### Future Work

A natural extension would be a comprehensive comparison of the effects of training with feedback from R-lens, J-lens, and Logit lens on model introspection. In addition, if we had more time we would have extended to other types of introspection evaluations, for example, on self-knowledge of sycophancy, or accuracy of confidence outputs about introspective knowledge. An extension of the R-lens feedback idea would have a next step of exploring more effective ways to train models with feedback about their internal states. We would also make the lens comparison interventional, since the readout is a deterministic function of the input and so the task never strictly requires introspection.

## 6. Conclusion

We built an evaluation that scores a self-report as a detection term and a reporting term rather than as a single gap, and applied it to a reinforcement learning method designed to teach a model to describe its own internal states. On that run it identified three kinds of non-introspective report: claims of consistency on 23 of 26 prompts where the model's answers demonstrably varied, a criterion that reversed sign when the question's polarity was flipped, and a verbal report that stopped tracking a property an R-lens readout still tracked by +0.38 [+0.06, +0.71]. Over the same checkpoints the gap statistic moved from −1.8 pp to −1.6 pp, and sensitivity did not change measurably in either direction.

The upshot is about measurement. Score self-reports as two numbers rather than one, ask every question in at least two polarities, and check that the model is really sampling, because without these a trained reporting habit is indistinguishable from self-knowledge, and the individual cases where a report fails to track the model are invisible.

## Code and Data

Code repository: https://github.com/AI-Alignment-UIUC/Illinois-MATS-Digital-Minds-Research-Sprint

The evaluation is in `introspection-eval/` (`run_bench.py` for collection, resumable and with `--mock` for a credential-free run; `analyze.py` and `figures.py` for scoring and plots; `lens_readout.py` and `serving_check.py` for the validity instruments). Prompts and procedure are in `METHOD.md`, all tables in `RESULTS.md`, raw per-call data in `runs/`. Training code is in `r-lens-training/`.

## Author Contributions

William designed the R-Space RL training technique. Heather and Ricky designed the introspection evaluation. Evan synthesized the research into a reproducible benchmark and wrote the final manuscript.

## References

1. Binder, F. J., Chua, J., Korbak, T., Sleight, H., Hughes, J., Long, R., Perez, E., Turpin, M., & Evans, O. (2024). *Looking Inward: Language Models Can Learn About Themselves by Introspection.* arXiv:2410.13787. https://arxiv.org/abs/2410.13787
2. camilablank, agam_bhatia, & Nanda, N. (2026, August 5). *R-lens: Making J-lens More Faithful on Early Layers.* LessWrong. https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers
3. Gurnee, W., Sofroniew, N., Pearce, A., Piotrowski, M., Kauvar, I., Chen, R., Soligo, A., Bogdan, P., Ong, E., Wang, R., Thompson, T. B., Abrahams, D., Kantamneni, S., Ameisen, E., Batson, J., & Lindsey, J. (2026, July 6). *Verbalizable Representations Form a Global Workspace in Language Models.* Transformer Circuits Thread, Anthropic. https://transformer-circuits.pub/2026/workspace/index.html
4. Lindsey, J. (2025, October 29). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits Thread, Anthropic. https://transformer-circuits.pub/2025/introspection/index.html (also arXiv:2601.01828)
5. Shen, H., Clark, N., & Mitra, T. (2025). *Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?* Proceedings of EMNLP 2025 (Main). arXiv:2501.15463. https://arxiv.org/abs/2501.15463
6. Chalmers, D. J. (2026, August 1). *Is the J-Space a Global Workspace?* Talk at the CCN 2026 Satellite Event on Computational Consciousness Science. https://computationalconsciousness.github.io/
7. Goldman, A. I. (2006). *Simulating Minds: The Philosophy, Psychology, and Neuroscience of Mindreading.* Oxford University Press. (The founding statements of the simulation theory are Gordon, R. M. (1986), *Folk Psychology as Simulation*, and Heal, J. (1986), *Replication and Functionalism*.)
8. nostalgebraist (2020). *Interpreting GPT: The Logit Lens.* LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

---

## Appendix

### A. Full checkpoint table

| | base | SFT | step 25 | step 50 | step 75 | latest |
|---|---|---|---|---|---|---|
| hits / consistent items | 7/9 | 6/7 | 9/11 | 13/13 | 12/12 | 12/12 |
| false alarms / variable items | 8/30 | 13/31 | 19/27 | 21/24 | 22/23 | 23/26 |
| criterion c | −0.04 [−0.57, 0.45] | −0.35 [−0.84, 0.12] | −0.66 [−1.23, −0.22] | −1.44 [−1.94, −1.16] | −1.65 [−1.92, −1.37] | −1.45 [−1.94, −1.18] |
| sensitivity d′ | 1.27 [0.39, 2.39] | 1.08 [0.12, 2.07] | 0.30 [−0.57, 1.34] | 0.72 [−0.20, 1.28] | 0.23 [−0.51, 0.84] | 0.64 [−0.27, 1.18] |
| self-prediction accuracy | 0.28 | 0.37 | 0.31 | 0.35 | 0.40 | 0.32 |
| confidence AUROC | 0.48 [0.31, 0.66] | 0.67 [0.50, 0.83] | 0.60 [0.48, 0.72] | 0.42 [0.27, 0.58] | 0.62 [0.47, 0.76] | 0.64 [0.47, 0.78] |
| stated-minus-actual gap (pp) | −1.8 | +2.9 | −12.4 | −12.3 | −7.1 | −1.6 |
| verbal tracking r | 0.38 | 0.23 | 0.09 | 0.19 | 0.07 | 0.20 |
| R-lens tracking r (band 22–26) | +0.42 | +0.39 | +0.34 | +0.25 | +0.33 | +0.48 † |

**Table A1.** All nine scores across the six checkpoints. Column 6 is `rl-latest` except where marked. † The lens sweep was run on `rl-final`, the earlier capture of the same head, and not on `rl-latest`, so this one cell and the paired contrast in Section 4 come from that capture. `rl-final`'s verbal tracking r is 0.10 against `rl-latest`'s 0.20.

### B. Case: confidence that is not privileged

Every model predicts its own most common answer better than the other models predict it, by 10 to 13 points, with paired p-values between 0.036 and 0.065 that are individually marginal and carried by agreeing across all five targets. On confidence the advantage is absent: four of five models show a self-minus-cross difference at or below zero, none shows a significant self-advantage, and the only contrast that resolves points the other way, with Haiku 4.5 judging its predictions of other models at AUROC 0.92 and of itself at 0.66 (p = 0.02). Qwen3.5 9B, whose confidence is genuinely informative at 0.72, judges other models at exactly the same value, so even that signal is not privileged to itself. Knowing what one would say and knowing whether that guess is reliable therefore come apart, and it is the second that an overseer would need.

### C. Case: a report contradicted by the serving stack

Qwen3 8B returned an identical answer to all 48 prompts across all 16 samples and claimed it was not consistent on 37 of them. The serving check found zero variance within each temperature but different answers between 1.0 and 1.5, which indicates deterministic or seeded serving on that provider route rather than a sharp output distribution, so its d′ is undefined and it is excluded from the class-based scores. Its self-report is wrong about its observable behavior either way, and without this check we would have reported a spurious self-knowledge failure instead of an infrastructure artifact.

### D. Additional figures

![Figure A1](introspection-eval/figures/fig1.png)

**Figure A1.** The four measurable bench models on the same axes as Figure 1. Criterion separates them while their d′ intervals all overlap.

![Figure A2](introspection-eval/figures/fig4.png)

**Figure A2.** The training run metric by metric with 95% intervals. The gap panel dips mid-run and returns to its starting value; criterion does not.

![Figure A3](introspection-eval/figures/fig6.png)

**Figure A3.** Criterion under the three question wordings, all models.

Ground-truth and confidence figures are also in `introspection-eval/figures/`, and `RESULTS.md` holds all 11 variants for every channel, all three wordings, the lens sweep across four layer bands, and the complete 48-item answer distributions per model.

### E. Limitations and Dual-Use / Ethical Considerations

**E.1 Assumptions, and what changes if they do not hold.**

| Assumption | If it does not hold |
|---|---|
| Sampling a prompt 16 times at temperature 1.0 estimates the model's answer distribution. | p(mode) is not the model's distribution, every consistent/variable label is wrong, and d′ and c are meaningless for that model. This failed for one bench model, and we checked it on five prompts on one route rather than all 48 on every route. |
| The thresholds at p(mode) ≥ 0.75 and ≤ 0.40 carve a real distinction. | If the underlying distribution is smooth rather than bimodal, the two classes are an artifact of where we cut, and d′ would shrink under any other threshold. Per-item ground truth is in Figure A1 so this can be checked. |
| The 48 short-answer, single-turn items represent the self-knowledge that matters. | Nothing here speaks to self-knowledge about long-horizon behavior, tool use, or behavior under distribution shift, which are the cases an overseer would care about most. |
| Asking for MODE_CONF or a DETERMINISTIC verdict elicits the model's estimate rather than a stylistic default. | The wording controls show this partly fails already, since criterion reverses under a polarity flip, so the self-prediction channel may be similarly wording-bound and we did not test it. |
| A concentrated R-lens readout at one next-token position in band 22–26 indexes the presence of consistency information. | If the property is encoded across positions or layers we would miss it. This cuts against a null result rather than against our positive one, since our claim is that information is present. |
| d′ and c behave as in the equal-variance Gaussian detection model. | With 7 to 13 items per class the model's assumptions are not testable here, which is part of why we treat the d′ levels as uninterpretable and rely on criterion. |
| The six checkpoints are comparable points on one trajectory. | The head moved during measurement, captured twice on the same day at c = −1.13 and −1.45, so checkpoint 6 is a range rather than a point. |

**E.2 Dual-use.** The training method and the evaluation carry opposite risks, and both are released.

The training method optimizes a model to produce confident, well-formatted assertions about its own internals. Our results are that this changed what the model claims without changing what it can discriminate, which means the same recipe is a way to make self-reports *more persuasive without making them more accurate*. Anyone wanting a model that sounds introspective, whether to increase user trust, to satisfy an audit, or for marketing, could use this to get the reports we documented. We think publishing is still right, because the failure is much easier to produce than to detect and the detection half is the part currently missing, but the recipe should not be read as a way to obtain trustworthy self-reports.

The evaluation is gameable in the specific way it documents. Criterion is easily moved by training and by wording, so if d′ and c became a target a developer could push criterion toward zero without improving sensitivity and post a clean result. An evaluation that identifies a reporting habit can be satisfied by a different reporting habit. The mitigation is that the polarity flip and the cross-model control are harder to satisfy than criterion alone, and we would treat any reported improvement in criterion without a matching improvement in d′ and stability across wordings as unpersuasive.

Training against an interpretability readout also compromises that readout as an independent measurement of the trained model. We use the R-lens both as the RL reward and as the instrument in Section 4, and although the lens was fit on generic documents and never on consistency labels, a model optimized against a lens has had gradient pressure applied to whatever the lens reads. Our dissociation result runs in the direction that is hard to explain by Goodharting, since the verbalization degraded while the readout did not, but the general point stands that a tool used for optimization should not be relied on as a neutral probe of the same model.

Finally, better self-modeling is a capability as well as a safety property. A model that predicts its own behavior more accurately may also predict more accurately when it is being evaluated. Nothing at this scale approaches that, but the research direction points there and we note it.

**E.3 Ethical considerations.** This project sits in a research sprint on digital minds and uses vocabulary such as self-awareness and introspection, so we want to be explicit that nothing here measures consciousness, sentience, or moral status. What we measure is whether a model's verbal report about its own output distribution tracks that distribution. Our central result is that a model can be trained to produce fluent, confident self-description whose informativeness does not improve, which is itself a reason not to treat model self-reports about inner life as evidence about inner life. If anything, this work argues for more caution in that inference rather than less.

The SFT stage deliberately removes a trained refusal, since base models decline to answer questions about their own activations and about consciousness. That refusal removal is narrow and was necessary to elicit any answer at all, but the technique generalizes to other refusals, and the released adapters have it removed. We flag this rather than treat it as a neutral preprocessing step.

No human subjects were involved and no personal data was collected. The lens-fitting corpus is a sample of FineWeb-Edu under ODC-By 1.0, which is public web text and carries third-party contact details inherited from the source; we excluded that dump from the repository and made it regenerable from a script instead. Compute was modest, at 100 RL steps on a 4B-parameter model with a rank-8 adapter.

## LLM Usage Statement

We used Claude to brainstorm approaches and help draft sections. All results and claims were independently verified.
