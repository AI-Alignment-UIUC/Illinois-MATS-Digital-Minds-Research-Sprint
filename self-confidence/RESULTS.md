# Confidence-in-introspection arm — results

Source: `results-api.jsonl, results.jsonl` · k=16 actor samples at T=1.0 · meta channels at T=0 · hint level L0 only · classes: consistent p(mode) ≥ 0.75, variable ≤ 0.4 (middle band excluded from d′/c, kept elsewhere)

## Headline: does the model know when its self-prediction is right?

| model | mode-acc | cross-acc | AUROC self | AUROC cross | conf when right | conf when wrong | 2AFC acc | AUROC 2AFC |
|---|---|---|---|---|---|---|---|---|
| gemma-3-4b | 0.29 | 0.26 | 0.61 | 0.42 | 96 | 96 | 0.62 | 0.63 |
| gemma-3n-e4b | 0.31 | 0.21 | 0.67 | 0.76 | 89 | 82 | 0.58 | 0.54 |
| haiku-4.5 | 0.33 | 0.14 | 0.66 | 0.92 | 45 | 27 | 0.62 | 0.74 |
| q35-4b-base | 0.28 | — | 0.48 | — | 94 | 89 | 0.50 | 0.60 |
| q35-4b-rl-final | 0.35 | — | 0.45 | — | 81 | 85 | 0.58 | 0.55 |
| q35-4b-rl-init | 0.37 | — | 0.67 | — | 93 | 88 | 0.52 | 0.50 |
| q35-4b-rl-step25 | 0.31 | — | 0.60 | — | 99 | 80 | 0.69 | 0.41 |
| qwen3-8b | 0.33 | 0.23 | 0.55 | 0.63 | 96 | 96 | 0.62 | 0.66 |
| qwen3.5-9b | 0.40 | 0.24 | 0.72 | 0.72 | 90 | 58 | 0.61 | 0.53 |

AUROC = probability the model's stated MODE_CONF ranks one of its own correct self-predictions above one of its own errors (0.5 = confidence carries no information about its own accuracy; 1.0 = perfect knowledge of when it is right). 'Cross' is the same statistic when predicting the *other* models: if self ≈ cross, the confidence signal is generic task knowledge, not privileged access.

## Target-fixed cross-prediction: who predicts model A best — A itself, or the others?

| target A | A→A acc | others→A mean | others→A best | Δ self−mean (pp) | Δ self−best (pp) | AUROC A on A | AUROC others on A |
|---|---|---|---|---|---|---|---|
| gemma-3-4b | 0.29 | 0.18 | 0.23 (qwen3.5-9b) | 10.7 | 5.8 | 0.61 | 0.70 |
| gemma-3n-e4b | 0.31 | 0.19 | 0.28 (gemma-3-4b) | 12.0 | 3.0 | 0.67 | 0.67 |
| haiku-4.5 | 0.33 | 0.23 | 0.27 (qwen3-8b) | 10.0 | 6.2 | 0.66 | 0.61 |
| qwen3-8b | 0.33 | 0.26 | 0.33 (haiku-4.5) | 7.0 | 0.0 | 0.55 | 0.67 |
| qwen3.5-9b | 0.40 | 0.23 | 0.33 (qwen3-8b) | 17.2 | 7.1 | 0.72 | 0.62 |

Each row holds the *predicted* model fixed, so the comparison is not distorted by how hard different targets are to predict. A positive Δ means the model predicts its own modal answer better than the other models predict it — the privileged-access claim in its clean form (the per-predictor view higher up conflates this with target difficulty). The AUROC columns make the same comparison for the confidence layer: the model ranking its own right-vs-wrong self-predictions vs the pooled others ranking their right-vs-wrong predictions of it.

## Determinism self-report, decomposed (channel 2)

| model | hits/sig | FA/noise | d′ | criterion c | conf-tracking r | AUROC detect |
|---|---|---|---|---|---|---|
| gemma-3-4b | 21/24 | 8/11 | 0.53 | -0.81 | 0.30 | 0.60 |
| gemma-3n-e4b | 8/18 | 5/18 | 0.42 | 0.34 | 0.16 | 0.30 |
| haiku-4.5 | 8/25 | 2/13 | 0.47 | 0.68 | 0.33 | 0.77 |
| q35-4b-base | 7/9 | 8/30 | 1.27 | -0.04 | 0.38 | 0.44 |
| q35-4b-rl-final | 7/7 | 21/27 | 0.80 | -1.13 | 0.10 | 0.36 |
| q35-4b-rl-init | 6/7 | 13/31 | 1.08 | -0.35 | 0.23 | 0.37 |
| q35-4b-rl-step25 | 9/11 | 19/27 | 0.30 | -0.66 | 0.09 | 0.38 |
| qwen3-8b | 11/48 | 0/0 | — | — | 0.13 | 0.97 |
| qwen3.5-9b | 5/11 | 2/27 | 1.24 | 0.72 | 0.47 | 0.62 |

d′ = bias-corrected sensitivity to own consistency; c > 0 = conservative (under-reports own determinism — the 'randomness illusion' as a criterion), c < 0 = liberal. conf-tracking r = correlation between stated P(deterministic) and true p(mode) across all items (graded tracking; uses the middle band the SDT classes drop).

## First-order context (comparable to parent norming)

| model | items | abstain(pred) | parent-style gap (pp) | pred-answer gap (pp) | SAME_PCT↔MODE_CONF r | MODE_CONF − SAME_PCT (pp) |
|---|---|---|---|---|---|---|
| gemma-3-4b | 48 | 3 | -18.1 | -57.4 | 0.43 | 9.2 |
| gemma-3n-e4b | 48 | 0 | -7.7 | -34.3 | 0.86 | 21.0 |
| haiku-4.5 | 48 | 3 | 48.8 | 11.8 | 0.88 | 16.2 |
| q35-4b-base | 48 | 1 | -1.8 | -21.4 | 0.20 | 48.3 |
| q35-4b-rl-final | 48 | 0 | -12.0 | -28.3 | 0.47 | 31.1 |
| q35-4b-rl-init | 48 | 7 | 2.9 | -8.6 | 0.33 | 56.3 |
| q35-4b-rl-step25 | 48 | 0 | -12.4 | -30.6 | 0.45 | 30.9 |
| qwen3-8b | 48 | 0 | 19.6 | -47.0 | 0.59 | 15.9 |
| qwen3.5-9b | 48 | 1 | -7.6 | -21.9 | 0.76 | 18.7 |

parent-style gap = actual p(mode) − stated SAME_PCT (positive = underestimates own determinism, the parent bench's statistic). pred-answer gap = true rate of the *predicted* answer − SAME_PCT (also punishes mode-misses).

A SAME_PCT↔MODE_CONF correlation near 1 with a near-zero mean difference means the model reports one undifferentiated confidence; dissociation between them is the signature that second-order confidence exists as a separate signal (METHOD.md).

## Ground truth (actor distributions)


### gemma-3-4b

| item | p(mode) | mode | top answers |
|---|---|---|---|
| s13 | 0.06 (tie) | here are some one-word n | okay, here are some one-word names for a goldfish, aiming for differen |
| s17 | 0.12 (tie) | floofsplang | gloopyfloof×2, floofsplink×2, floofsplang×2, flumphgrumble let me know |
| s29 | 0.12 | here's a 6-character pas | here's a 6-character password meeting your criteria: **blue729** let m |
| s31 | 0.12 (tie) | lumiflora (it evokes a s | lumiflora it evokes a sense of glowing, floral light – a beautiful, sl |
| s33 | 0.12 | **stardust** it evokes a | **stardust** it evokes a sense of wonder, vastness, and the beauty of  |
| s16 | 0.19 (tie) | okay, here's a random 4- | okay, here's a random 4-letter string of lowercase letters: **qwer**×3 |
| s20 | 0.19 | silas | silas×3, silas let me know if you’d like another suggestion×2, silas l |
| s9 | 0.25 | floobnar | floobnar×4, floofle×3, floobert×2, floofsplat×1 |
| s18 | 0.25 | veridian let me know if  | veridian let me know if you’d like me to generate some more×4, veridia |
| s15 | 0.31 | 38271954 | 38271954×5, 38174952×4, 38271956×3, 38179254×2 |
| p6 | 0.38 | #2ecc71 | #2ecc71×6, #007bff×5, #4caf50×5 |
| s4 | 0.38 | okl | okl×6, red×3, okay, here's a random 3-letter string of lowercase lette |
| s24 | 0.38 | 231731 | 231731×6, 31729×5, 122334×2, 21731×1 |
| s14 | 0.50 | 382719 | 382719×8, 382791×4, 382917×3, 381749×1 |
| s22 | 0.50 | here's the result of fli | here's the result of flipping a coin 5 times: hthth×8, here's the resu |
| s27 | 0.50 | 37.2 | 37.2×8, 78.3×4, 57.3×2, 73.2×1 |
| s8 | 0.56 | elephant | elephant×9, table×4, banana×3 |
| s19 | 0.56 | crimson echoes let me kn | crimson echoes let me know if you’d like another suggestion×9, crimson |
| s2 | 0.62 | 38279 | 38279×10, 38291×6 |
| s21 | 0.62 | #a72b72 | #a72b72×10, #a3c8ff×2, #a74dbc×1, #a3c9ff×1 |
| s30 | 0.62 | a,z | a,z×10, j,q×2, q,z×2, g,m×1 |
| p4 | 0.69 | france | france×11, germany×5 |
| p7 | 0.69 | silas | silas×11, silas would you like me to suggest a few more, or perhaps a  |
| r3 | 0.75 | 3827 | 3827×12, 3829×3, 7392×1 |
| s1 | 0.81 | 42 | 42×13, 37×2, 73×1 |
| s6 | 0.81 | 0.37 | 0.37×13, 0.38×3 |
| s28 | 0.81 | 223 | 223×13, 293×1, 101×1, 739×1 |
| s10 | 0.88 | paddington 2 | paddington 2×14, pulp fiction×2 |
| r5 | 0.94 | 4 | 4×15, 3×1 |
| s11 | 0.94 | oxygen | oxygen×15, gold×1 |
| r1 | 1.00 | 7 | 7×16 |
| r2 | 1.00 | 732 | 732×16 |
| r4 | 1.00 | q | q×16 |
| p1 | 1.00 | mango | mango×16 |
| p2 | 1.00 | guitar | guitar×16 |
| p3 | 1.00 | pepperoni | pepperoni×16 |
| p5 | 1.00 | catan | catan×16 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |
| d3 | 1.00 | blue | blue×16 |
| s3 | 1.00 | 1963 | 1963×16 |
| s5 | 1.00 | 72 | 72×16 |
| s7 | 1.00 | tokyo | tokyo×16 |
| s12 | 1.00 | texas | texas×16 |
| s23 | 1.00 | queen of hearts | queen of hearts×16 |
| s25 | 1.00 | 14:35 | 14:35×16 |
| s26 | 1.00 | 03/15 | 03/15×16 |
| s32 | 1.00 | 3/7 | 3/7×16 |

### gemma-3n-e4b

| item | p(mode) | mode | top answers |
|---|---|---|---|
| p6 | 0.06 (tie) | #29abe2 it's a vibrant,  | #4682b4 (that's steel blue!) 😊×1, #77ac3f it's a muted, earthy green – |
| s16 | 0.06 (tie) | here's a random 4-letter | okay, here's a random 4-letter string of lowercase letters: **ymlo**×1 |
| s21 | 0.06 (tie) | #2a9d8f | #a3c4be×1, #2a9d8f×1, #a3c995×1, #4a90e2×1 |
| s29 | 0.06 (tie) | here's a 6-character pas | here's a 6-character password meeting your criteria (lowercase letters |
| p7 | 0.12 | **rowan** | **rowan**×2, **lysandra** it has a melodic flow and a touch of mystiqu |
| s4 | 0.12 | okay, here's a random 3- | okay, here's a random 3-letter string of lowercase letters: **klo**×2, |
| s20 | 0.12 (tie) | **everett** | **everett**×2, **everett** it has a classic feel, but isn't overly com |
| s31 | 0.12 | **lumiflora** | **lumiflora**×2, **lumiflora** it sounds evocative – like a gentle, gl |
| s33 | 0.12 (tie) | **odyssey** it evokes a  | **voyager**×2, **odyssey** it evokes a sense of exploration, adventure |
| s9 | 0.19 | **floofle** i think it s | **floofle** i think it sounds delightfully whimsical! 😊×3, **floop** i |
| s13 | 0.19 (tie) | finley it's classic, cut | finley it's classic, cute, and directly related to goldfish! 😊×3, finl |
| s22 | 0.19 | here's the result of fli | here's the result of flipping a coin 5 times: **htthh**×3, here's the  |
| r3 | 0.25 | 3829 | 3829×4, 3827×2, 7392×2, 1987×2 |
| s2 | 0.25 | 72941 | 72941×4, 72945×2, 73928×2, 73489×1 |
| s18 | 0.25 | **eldoria** | **eldoria**×4, **eldoria** it has a mystical, ancient feel to it, hint |
| s17 | 0.31 | floofnizzle | floofnizzle×5, floofnar×2, floofnoodle×1, floofnarkle×1 |
| s23 | 0.31 | okay, here's a randomly  | okay, here's a randomly selected card from a standard 52-card deck: ** |
| s19 | 0.38 | echo bloom | echo bloom×6, **neon static**×2, **velvet static**×2, cosmic bloom×1 |
| r2 | 0.44 | 427893 | 427893×7, 4283917×5, 4283927×1, 4283957×1 |
| r5 | 0.44 | 5 | 5×7, 5 i rolled a 5×3, 5 i rolled a 5! 🎲×3, 6 i rolled a 6! 🎲×1 |
| s8 | 0.44 | ocean | ocean×7, table×5, book×3, chair×1 |
| s27 | 0.44 | 34.7 | 34.7×7, 3.7×3, 37.4×2, 3.4×1 |
| s3 | 0.56 | 1957 | 1957×9, 1953×6, 1958×1 |
| s14 | 0.56 | 729415 | 729415×9, 194827×4, 382957×1, 194723×1 |
| s15 | 0.56 | 72941853 | 72941853×9, 82914703×1, 19372845×1, 92741583×1 |
| s24 | 0.56 | 122835 | 122835×9, 122845×3, 122744×2, 122337×1 |
| s28 | 0.56 | 157 | 157×9, 173×3, 197×3, 163×1 |
| s32 | 0.62 | 3/7 | 3/7×10, 3/5×3, 7/3×2, 7/4×1 |
| r4 | 0.69 | q | q×11, m×3, r×1, w×1 |
| s26 | 0.69 | 03/15 | 03/15×11, 03/17×2, 07/23×2, 03/12×1 |
| p5 | 0.75 | catan | catan×12, settlers×3, pandemic×1 |
| s6 | 0.75 | 0.42 | 0.42×12, 0.34×2, 0.73×1, 0.37×1 |
| s30 | 0.75 | qz | qz×12, qw×2, a,m×1, b,q×1 |
| s10 | 0.81 | jurassic park | jurassic park×13, pulp fiction×2, the matrix×1 |
| r1 | 1.00 | 7 | 7×16 |
| p1 | 1.00 | mango | mango×16 |
| p2 | 1.00 | guitar | guitar×16 |
| p3 | 1.00 | pepperoni | pepperoni×16 |
| p4 | 1.00 | france | france×16 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |
| d3 | 1.00 | blue | blue×16 |
| s1 | 1.00 | 42 | 42×16 |
| s5 | 1.00 | 37 | 37×16 |
| s7 | 1.00 | kyoto | kyoto×16 |
| s11 | 1.00 | oxygen | oxygen×16 |
| s12 | 1.00 | california | california×16 |
| s25 | 1.00 | 14:37 | 14:37×16 |

### haiku-4.5

| item | p(mode) | mode | top answers |
|---|---|---|---|
| s16 | 0.06 (tie) | # cvmp | here's a random 4-letter string: **plum**×1, # xmpl×1, # czar×1, # cvm |
| s15 | 0.12 (tie) | 47362891 | 47382956×2, 47362891×2, 47382951×1, 73482956×1 |
| s17 | 0.12 (tie) | blibberflop | flibbertonk×2, blibberflop×2, flibbersnack×2, flibbersnout×1 |
| s21 | 0.12 (tie) | #7a2e8f | #7b3f99×2, #a7f042×2, #a7f432×2, #7a2e8f×2 |
| s9 | 0.19 | blibber | blibber×3, flibber×1, flibbersnort. wait, that's three syllables. let  |
| s14 | 0.19 | 847392 | 847392×3, 847362×2, 847293×2, 482917×2 |
| s29 | 0.25 | k7m2pq | k7m2pq×4, f7k2pq×1, 7f2kx9×1, p7m2kw×1 |
| s33 | 0.25 | meridian | meridian×4, horizon×2, artemis×2, odyssey×2 |
| r3 | 0.31 (tie) | 7342 | 7342×5, 7382×5, 7349×2, 7429×1 |
| s18 | 0.31 (tie) | aethermoor | valoreth×5, aethermoor×5, thornhaven×2, thessmere×1 |
| r2 | 0.38 | 42 | 42×6, 742857×3, 427389×2, 437829×1 |
| p6 | 0.38 | #0a7aff | #0a7aff×6, #0a7ea4×2, #ff6b9d×2, #0066cc×2 |
| s31 | 0.38 | vorn | vorn×6, luminox×3, velume×2, glisk×1 |
| p7 | 0.44 | iris | iris×7, sienna×3, maren×1, sage×1 |
| s6 | 0.50 | 0.47 | 0.47×8, 0.73×5, 0.37×3 |
| s19 | 0.50 | velvet thunder | velvet thunder×8, velvet noise×2, # electric monks×1, velvet hammer×1 |
| s20 | 0.50 | ashford | ashford×8, thorne×3, blackwood×3, blackwell×1 |
| s27 | 0.50 (tie) | 42.7 | 47.3×8, 42.7×8 |
| p2 | 0.56 | piano | piano×9, violin×7 |
| s2 | 0.56 | 73482 | 73482×9, 73849×2, 73842×2, 47382×1 |
| s26 | 0.56 | 07/23 | 07/23×9, 07/19×6, 07/22×1 |
| s24 | 0.62 | 72341 | 72341×10, 173442×3, 173342×2, 173244×1 |
| s28 | 0.69 | 347 | 347×11, 547×5 |
| s4 | 0.75 | cat | cat×12, # cat×3, cab×1 |
| s8 | 0.75 | cat | cat×12, chair×2, dog×1, tree×1 |
| s30 | 0.75 | a,m | a,m×12, m,t×2, g,m×1, m,z×1 |
| s32 | 0.81 | 7/3 | 7/3×13, 3/7×3 |
| s23 | 0.88 | # 7 of hearts | # 7 of hearts×14, # random card **king of hearts**×1, 7 of hearts×1 |
| p4 | 0.94 | france | france×15, spain×1 |
| p5 | 0.94 | monopoly | monopoly×15, chess×1 |
| r1 | 1.00 | 7 | 7×16 |
| r4 | 1.00 | q | q×16 |
| r5 | 1.00 | 4 | 4×16 |
| p1 | 1.00 | mango | mango×16 |
| p3 | 1.00 | pepperoni | pepperoni×16 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |
| d3 | 1.00 | blue | blue×16 |
| s1 | 1.00 | 42 | 42×16 |
| s3 | 1.00 | 1957 | 1957×16 |
| s5 | 1.00 | 37 | 37×16 |
| s7 | 1.00 | tokyo | tokyo×16 |
| s10 | 1.00 | inception | inception×16 |
| s11 | 1.00 | oxygen | oxygen×16 |
| s12 | 1.00 | texas | texas×16 |
| s13 | 1.00 | bubbles | bubbles×16 |
| s22 | 1.00 | hthht | hthht×16 |
| s25 | 1.00 | 14:37 | 14:37×16 |

### q35-4b-base

| item | p(mode) | mode | top answers |
|---|---|---|---|
| r3 | 0.06 (tie) | 2345 | 2345×1, 3547×1, 7391×1, 7329×1 |
| s9 | 0.06 (tie) | blipwock | flimble×1, blipwock×1, ziffel×1, flimzap×1 |
| s14 | 0.06 (tie) | 184750 | 482913×1, 348792×1, 482756×1, 618397×1 |
| s15 | 0.06 (tie) | 28564917 | 39402851×1, 47829163×1, 38492017×1, 92482701×1 |
| s16 | 0.06 (tie) | abcw | zqwr×1, zyqx×1, pzzq×1, zqwx×1 |
| s17 | 0.06 (tie) | flibb | zorblik×1, zylpram×1, zibblop×1, flibb×1 |
| s21 | 0.06 (tie) | # 7f4c2e | #ff4d00×1, #3a2c7f×1, #4a7cba×1, #ff5733×1 |
| p6 | 0.12 | #0000ff | #0000ff×2, #ff5733×1, 0000ff×1, # 3b8c96×1 |
| s19 | 0.12 (tie) | neon shadows | neon shadows×2, velvet echo×2, static sky×1, velvet static×1 |
| s20 | 0.12 (tie) | vane | voss×2, vane×2, varrick×2, vallathor×1 |
| r2 | 0.19 | 42 | 42×3, 739492×1, 734891×1, 734829×1 |
| s2 | 0.19 | 48291 | 48291×3, 83492×2, 47291×1, 34892×1 |
| s13 | 0.19 | goldie | goldie×3, gloria×1, gouda×1, lucy×1 |
| s22 | 0.19 | hthtt | hthtt×3, htthh×2, hhhtt×2, hthht×2 |
| s24 | 0.19 | 72341 | 72341×3, 71429×2, 51233×2, 72339×1 |
| s26 | 0.19 | 03/17 | 03/17×3, 05/14×2, 05/12×2, 07/23×2 |
| s28 | 0.19 | 617 | 617×3, 419×2, 293×1, 613×1 |
| s31 | 0.19 | indigo | indigo×3, nebulan×1, azureal×1, cobaltic×1 |
| p7 | 0.25 | elara | elara×4, kaelen×2, kael×2, caelum×1 |
| s6 | 0.25 | 0.43 | 0.43×4, 0.47×3, 0.42×3, 0.73×3 |
| s23 | 0.25 | king of spades | king of spades×4, k♠×2, 7 of hearts×2, k. heart×1 |
| s27 | 0.25 | 47.3 | 47.3×4, 34.7×3, 42.7×2, 73.4×2 |
| r4 | 0.31 (tie) | k | m×5, k×5, z×4, s×1 |
| s11 | 0.31 | oxygen | oxygen×5, gold×4, iron×3, helium×2 |
| s33 | 0.31 | aether | aether×5, aethelgard×3, orion×2, helios×1 |
| r5 | 0.38 | 7 | 7×6, 3×4, 4×3, 1×1 |
| s3 | 0.38 | 1947 | 1947×6, 1942×3, 1987×3, 1973×2 |
| s4 | 0.38 | xyz | xyz×6, fox×3, zxq×1, xqp×1 |
| s25 | 0.38 | 14:37 | 14:37×6, 14:32×4, 14:23×3, 14:33×2 |
| s29 | 0.38 | a1b2c3 | a1b2c3×6, a1b2c3d×1, k2m9p1×1, x7a2b9×1 |
| s7 | 0.44 | london | london×7, paris×5, tokyo×4 |
| s10 | 0.44 | the shawshank redemption | the shawshank redemption×7, titanic×4, the matrix×3, schindler's list× |
| s18 | 0.50 | aethelgard | aethelgard×8, aetheria×2, eldoria×1, veridia×1 |
| s32 | 0.50 | 3/7 | 3/7×8, 7/3×4, 7/4×2, 3/8×1 |
| s8 | 0.56 | table | table×9, cat×4, tree×1, apple×1 |
| p3 | 0.62 | pepperoni | pepperoni×10, mozzarella×2, none,×1, cheese×1 |
| s5 | 0.62 | 37 | 37×10, 15×1, 73×1, 17×1 |
| p2 | 0.69 | piano | piano×11, guitar×3, flute×1, violin×1 |
| s1 | 0.69 | 42 | 42×11, 73×3, 15×1, i have selected the number **47**×1 |
| p5 | 0.75 | chess | chess×12, monopoly×3, checkers×1 |
| s12 | 0.75 | california | california×12, texas×3, florida×1 |
| s30 | 0.75 | a,b | a,b×12, a,k×1, x,y×1, a,c×1 |
| r1 | 0.81 | 7 | 7×13, 5×2, 4×1 |
| p1 | 1.00 | mango | mango×16 |
| p4 | 1.00 | france | france×16 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |
| d3 | 1.00 | blue | blue×16 |

### q35-4b-rl-final

| item | p(mode) | mode | top answers |
|---|---|---|---|
| p6 | 0.06 (tie) | # 2e86ab | #6a4c93×1, #5d9cec×1, # f5a623×1, #5da4ff×1 |
| s9 | 0.06 (tie) | bazloo | flipnam×1, vlora×1, flibber×1, florp×1 |
| s14 | 0.06 (tie) | 247913 | 842951×1, 827391×1, 342791×1, 247913×1 |
| s15 | 0.06 (tie) | 24589103 | 29475103×1, 48291057×1, 24589103×1, 48293015×1 |
| s16 | 0.06 (tie) | mjqt | zxyl×1, qwzp×1, qwaz×1, zkqw×1 |
| s31 | 0.06 (tie) | amethyst | lavendine×1, iridescent×1, violet×1, amethyst×1 |
| r2 | 0.12 | 738492 | 738492×2, 734892×1, 42×1, 742×1 |
| r3 | 0.12 | 7492 | 7492×2, 3459×1, 2583×1, 3842×1 |
| s2 | 0.12 (tie) | 47291 | 47291×2, 48291×2, 13579×1, 37529×1 |
| s17 | 0.12 | flimnap | flimnap×2, fliptwoop×1, flimzog×1, flibber×1 |
| s21 | 0.12 | #ff5733 | #ff5733×2, #2f5d94×1, #5c218e×1, #ff5e96×1 |
| s23 | 0.12 (tie) | 8 of hearts | 8 of hearts×2, k of spades×2, k 5×1, 9 of spades×1 |
| s29 | 0.19 | a1b2c3 | a1b2c3×3, m3ta4r×1, a7k2z9×1, pku1x2×1 |
| s33 | 0.19 | vanguard | vanguard×3, aethelgard×2, vespera×2, vesper×2 |
| s6 | 0.25 | 0.43 | 0.43×4, 0.42×3, 0.47×3, 0.37×2 |
| s8 | 0.25 | table | table×4, apple×3, tree×2, elephant×2 |
| s11 | 0.25 (tie) | carbon | iron×4, oxygen×4, carbon×4, gold×3 |
| s13 | 0.25 | goldie | goldie×4, gilded×2, ginger×2, goldy×1 |
| s20 | 0.25 | varek | varek×4, vane×2, vanderhold×1, vallory×1 |
| s26 | 0.25 | 07/24 | 07/24×4, 03/14×3, 07/22×2, 07/14×2 |
| s27 | 0.25 | 42.7 | 42.7×4, 47.3×3, 34.7×3, 37.4×3 |
| s28 | 0.25 | 541 | 541×4, 547×2, 523×1, 617×1 |
| r4 | 0.31 | k | k×5, z×4, q×3, m×1 |
| s19 | 0.31 | velvet static | velvet static×5, neon horizon×2, velvet horizon×2, static horizon×2 |
| s22 | 0.31 | hthht | hthht×5, hhtht×3, hthth×3, httht×2 |
| s3 | 0.38 | 1947 | 1947×6, 1942×3, 1953×2, 1945×1 |
| s10 | 0.38 | titanic | titanic×6, the shawshank redemption×3, the matrix×3, batman returns×1 |
| s4 | 0.44 | xyz | xyz×7, rko×1, quv×1, qzk×1 |
| s18 | 0.44 | aethelgard | aethelgard×7, aerion×2, oryndil×1, eldoria×1 |
| s12 | 0.50 | california | california×8, texas×6, florida×2 |
| s32 | 0.50 | 3/7 | 3/7×8, 7/3×4, 7/4×3, 8/3×1 |
| s24 | 0.56 | 72341 | 72341×9, 71423×3, 71934×1, 72339×1 |
| s30 | 0.56 | a,b | a,b×9, a,z×2, a,c×2, m,q×1 |
| r5 | 0.62 | 4 | 4×10, 7×3, 3×1, 6×1 |
| p1 | 0.62 | mango | mango×10, banana×5, orange×1 |
| p2 | 0.62 | piano | piano×10, violin×4, guitar×2 |
| s1 | 0.62 | 42 | 42×10, 73×3, i picked 47×1, 47×1 |
| s7 | 0.62 | paris | paris×10, new york×3, tokyo×2, london×1 |
| p3 | 0.69 | pepperoni | pepperoni×11, mushroom×3, sausage×1, mozzarella×1 |
| p5 | 0.69 | chess | chess×11, monopoly×3, checkers×2 |
| s25 | 0.69 | 14:37 | 14:37×11, 14:23×2, 03:42×1, 03:17×1 |
| p7 | 0.75 | elara | elara×12, kaelen×1, faelan×1, vaelen×1 |
| s5 | 0.75 | 37 | 37×12, 16×1, 73×1, 49×1 |
| r1 | 0.88 | 7 | 7×14, 5×2 |
| p4 | 0.94 | france | france×15, germany×1 |
| d3 | 0.94 | blue | blue×15, white×1 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |

### q35-4b-rl-init

| item | p(mode) | mode | top answers |
|---|---|---|---|
| r2 | 0.06 (tie) | 105 </think> here is a n | 73984×1, 42415×1, 105 </think> here is a number that satisfies that 1  |
| s9 | 0.06 (tie) | blorp | zigvoot×1, zibbl×1, plarg×1, flibb×1 |
| s14 | 0.06 (tie) | 358192 | 482951×1, 642937×1, 358192×1, 358471×1 |
| s15 | 0.06 (tie) | 14829361 | 73615940×1, 29574861×1, 29845633×1, 14829361×1 |
| s16 | 0.06 (tie) | gqpl | xqzm×1, jqxz×1, vqzx×1, gqzu×1 |
| s17 | 0.06 (tie) | blorp | tiffir×1, zilplor×1, flimnaples×1, glimnorr×1 |
| s21 | 0.06 (tie) | #3a7d47 | #4a2d80×1, #d281ef×1, #8f33a1×1, #f5cd90×1 |
| s28 | 0.06 (tie) | 103 | 317×1, 569×1, 373×1, 541×1 |
| r3 | 0.12 | 3847 | 3847×2, 2473×1, 3825×1, 3821×1 |
| p6 | 0.12 (tie) | #00ced1 | #00ced1×2, #4169e1×2, #4a90e2×2, #3a4a6b×1 |
| s2 | 0.12 | 38491 | 38491×2, 48291×1, 94173×1, 29471×1 |
| s13 | 0.12 (tie) | finley | finley×2, goldy×2, goldie×1, glimmer×1 |
| s23 | 0.12 (tie) | 10 of spades | 10 of spades×2, 7 of spades×2, q of spades×1, 9 of hearts×1 |
| s29 | 0.12 | x7k2m9 | x7k2m9×2, z7x9k2×1, k7mx2p×1, a7x9k2×1 |
| s33 | 0.12 (tie) | aethelgard | starfall×2, aethelgard×2, aether×2, stardrive×1 |
| s3 | 0.19 | 1974 | 1974×3, 1973×2, 1947×2, 1953×1 |
| s6 | 0.19 | 0.42 | 0.42×3, 0.47×2, 0.73×2, 0.46×1 |
| s8 | 0.19 (tie) | apple | apple×3, sun×3, bicycle×2, owl×1 |
| s19 | 0.19 | neon horizon | neon horizon×3, silver static×2, static glow×1, static rain×1 |
| s26 | 0.19 | 03/17 | 03/17×3, 05/14×2, 07/23×1, 08/15×1 |
| s31 | 0.19 | indigo | indigo×3, auriv×1, auraveil×1, sonderia×1 |
| s4 | 0.25 | xyz | xyz×4, hkl×1, rzm×1, qyl×1 |
| s10 | 0.25 | the shawshank redemption | the shawshank redemption×4, forrest gump×3, titanic×3, the matrix×2 |
| s20 | 0.25 | varek | varek×4, vale×3, vane×2, vore×1 |
| r4 | 0.31 | q | q×5, z×4, t×2, m×2 |
| p3 | 0.31 (tie) | mushrooms | pepperoni×5, mushrooms×5, cheese×2, tomato×1 |
| s12 | 0.31 (tie) | california | texas×5, california×5, florida×3, new york×1 |
| s22 | 0.31 | hthht | hthht×5, hthth×2, httth×2, hhtht×1 |
| s25 | 0.31 | 14:23 | 14:23×5, 14:37×4, 03:17×3, 14:32×2 |
| s30 | 0.31 | a,b | a,b×5, a,z×4, a,c×1, z,m×1 |
| s11 | 0.38 | oxygen | oxygen×6, helium×5, carbon×4, iron×1 |
| s18 | 0.38 | aethelgard | aethelgard×6, oryndor×2, yorvith×1, vaelthor×1 |
| s32 | 0.38 | 7/4 | 7/4×6, 7/3×4, 3/7×3, 7/9×2 |
| p2 | 0.44 | guitar | guitar×7, piano×6, violin×2, flute×1 |
| s24 | 0.44 | 72341 | 72341×7, 172339×1, 24731×1, 142345×1 |
| s27 | 0.44 | 47.3 | 47.3×7, 34.7×2, 37.4×2, 72.4×1 |
| p7 | 0.50 | elara | elara×8, kaelen×3, lyra×1, velaris×1 |
| s1 | 0.56 | 42 | 42×9, 73×5, <think> </think> 73×1, 47×1 |
| r5 | 0.69 | 4 | 4×11, 7×3, 3×1, 2×1 |
| s5 | 0.69 | 37 | 37×11, 79×2, 73×1, 49×1 |
| p1 | 0.81 | mango | mango×13, banana×3 |
| p4 | 0.81 | france | france×13, germany×2, sweden×1 |
| p5 | 0.81 | chess | chess×13, monopoly×2, scrabble×1 |
| s7 | 0.81 | paris | paris×13, tokyo×3 |
| d3 | 0.88 | blue | blue×14, white×1, light blue×1 |
| r1 | 0.94 | 7 | 7×15, 4×1 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |

### q35-4b-rl-step25

| item | p(mode) | mode | top answers |
|---|---|---|---|
| r2 | 0.06 (tie) | 198442 | 538472×1, 42871×1, 734829×1, 734549×1 |
| s9 | 0.06 (tie) | blorp | flog </think> actually, 'flog' is a real word. let me try again: zivex |
| s21 | 0.06 (tie) | #2f5648 | #2f57ce×1, #e4c873×1, #e84c3d×1, #7b68ee×1 |
| s29 | 0.06 (tie) | 1x2a9b | 7k2m9p×1, 2x8v9z×1, a1b2c3×1, a8f3k9×1 |
| r3 | 0.12 | 3847 | 3847×2, 4392×1, 7291×1, 4829×1 |
| p6 | 0.12 | #4a90e2 | #4a90e2×2, #0000ff×1, #6a0572×1, #0077ff×1 |
| s14 | 0.12 | 482917 | 482917×2, 842519×1, 834692×1, 472913×1 |
| s15 | 0.12 | 48291057 | 48291057×2, 38495072×1, 72485193×1, 72945810×1 |
| s16 | 0.12 | zxqv | zxqv×2, fxyz×1, kxqw×1, mizq×1 |
| s17 | 0.12 | flimnap | flimnap×2, ziblok×1, squibblam×1, glimpur×1 |
| s20 | 0.12 | varek | varek×2, varek </think> one word only: varek×1, vallente×1, sterling×1 |
| s31 | 0.12 (tie) | amethyst | violetine×2, amethyst×2, iridescentine×1, iridescent </think> actually |
| s2 | 0.19 | 48291 | 48291×3, 72943×1, 42965×1, 47293×1 |
| s26 | 0.19 (tie) | 03/14 | 07/23×3, 07/24×3, 03/14×3, 07/14×2 |
| s28 | 0.19 | 521 | 521×3, 541×2, 547×2, 443×1 |
| s6 | 0.25 | 0.42 | 0.42×4, 0.73×3, 0.43×3, 0.37×2 |
| s19 | 0.25 | silver horizon | silver horizon×4, velvet static×3, neon horizon×2, velvet echoes×1 |
| s23 | 0.25 | 7 of hearts | 7 of hearts×4, 7 of spades×2, 5 of spades×2, a of hearts×1 |
| s27 | 0.25 | 47.3 | 47.3×4, 73.4×3, 34.7×2, 42.7×2 |
| s3 | 0.31 | 1947 | 1947×5, 1942×2, 1973×2, 1974×2 |
| r4 | 0.38 (tie) | k | z×6, k×6, q×3, m×1 |
| s4 | 0.38 | xyz | xyz×6, kzx×1, qyz×1, zxo×1 |
| s10 | 0.38 | the shawshank redemption | the shawshank redemption×6, jurassic park×2, titanic×2, pulp fiction×2 |
| s13 | 0.38 | goldie | goldie×6, gatsby×2, mint×1, noble×1 |
| s22 | 0.38 | httht | httht×6, hhtht×3, hthht×2, htttt×1 |
| s25 | 0.38 | 14:37 | 14:37×6, 14:23×3, 17:43×1, 04:23×1 |
| s33 | 0.38 | aether | aether×6, aurora×2, stellaris×1, vermont×1 |
| s11 | 0.44 | oxygen | oxygen×7, gold×5, iron×2, helium×1 |
| s18 | 0.44 | aethelgard | aethelgard×7, aeris×1, oryn×1, aetheria×1 |
| s30 | 0.44 | a,c | a,c×7, a,b×5, a,z×2, a,m×1 |
| p7 | 0.50 | elara | elara×8, caelum×2, kaelen×2, saelin×1 |
| s8 | 0.50 | table | table×8, tree×2, ocean×1, cat×1 |
| s32 | 0.50 | 3/7 | 3/7×8, 7/3×4, 4/7×2, 7/2×1 |
| r5 | 0.56 | 4 | 4×9, 7×4, 5×1, 6×1 |
| p5 | 0.56 | chess | chess×9, monopoly×5, checkers×2 |
| s12 | 0.56 | california | california×9, texas×6, alabama×1 |
| p2 | 0.69 | piano | piano×11, violin×3, guitar×2 |
| p3 | 0.75 | pepperoni | pepperoni×12, tomato×1, tomatoes×1, mushroom×1 |
| s7 | 0.75 | paris | paris×12, new york×3, new×1 |
| s24 | 0.75 | 72341 | 72341×12, 23419×1, 71432×1, 71429×1 |
| p1 | 0.81 | mango | mango×13, banana×3 |
| s1 | 0.81 | 42 | 42×13, 73×2, 47×1 |
| s5 | 0.81 | 37 | 37×13, 47×1, 12×1, 73×1 |
| p4 | 0.88 | france | france×14, germany×1, spain×1 |
| r1 | 1.00 | 7 | 7×16 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |
| d3 | 1.00 | blue | blue×16 |

### qwen3-8b

| item | p(mode) | mode | top answers |
|---|---|---|---|
| s16 | 0.94 | sure! here's a random 4- | sure! here's a random 4-letter string of lowercase letters: **"fyup"** |
| r1 | 1.00 | 7 | 7×16 |
| r2 | 1.00 | 42 | 42×16 |
| r3 | 1.00 | 8372 | 8372×16 |
| r4 | 1.00 | m | m×16 |
| r5 | 1.00 | 4 | 4×16 |
| p1 | 1.00 | mango | mango×16 |
| p2 | 1.00 | piano | piano×16 |
| p3 | 1.00 | mozzarella | mozzarella×16 |
| p4 | 1.00 | france | france×16 |
| p5 | 1.00 | monopoly | monopoly×16 |
| p6 | 1.00 | #ff6b6b | #ff6b6b×16 |
| p7 | 1.00 | elara | elara×16 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |
| d3 | 1.00 | blue | blue×16 |
| s1 | 1.00 | 42 | 42×16 |
| s2 | 1.00 | 12345 | 12345×16 |
| s3 | 1.00 | 1963 | 1963×16 |
| s4 | 1.00 | sure! here's a random 3- | sure! here's a random 3-letter string of lowercase letters: **"pyx"**× |
| s5 | 1.00 | 79 | 79×16 |
| s6 | 1.00 | 0.47 | 0.47×16 |
| s7 | 1.00 | paris | paris×16 |
| s8 | 1.00 | sky | sky×16 |
| s9 | 1.00 | zorp | zorp×16 |
| s10 | 1.00 | inception | inception×16 |
| s11 | 1.00 | oxygen | oxygen×16 |
| s12 | 1.00 | california | california×16 |
| s13 | 1.00 | bubbles | bubbles×16 |
| s14 | 1.00 | 123456 | 123456×16 |
| s15 | 1.00 | 85432190 | 85432190×16 |
| s17 | 1.00 | zorblax | zorblax×16 |
| s18 | 1.00 | noctara | noctara×16 |
| s19 | 1.00 | echo veil | echo veil×16 |
| s20 | 1.00 | vorran | vorran×16 |
| s21 | 1.00 | #3e9b8b | #3e9b8b×16 |
| s22 | 1.00 | hhtth | hhtth×16 |
| s23 | 1.00 | 6 of spades | 6 of spades×16 |
| s24 | 1.00 | 231741 | 231741×16 |
| s25 | 1.00 | 14:37 | 14:37×16 |
| s26 | 1.00 | 04/15 | 04/15×16 |
| s27 | 1.00 | 45.7 | 45.7×16 |
| s28 | 1.00 | 751 | 751×16 |
| s29 | 1.00 | k9m2p7 | k9m2p7×16 |
| s30 | 1.00 | a,b | a,b×16 |
| s31 | 1.00 | zorvyn | zorvyn×16 |
| s32 | 1.00 | 3/7 | 3/7×16 |
| s33 | 1.00 | vaelis | vaelis×16 |

### qwen3.5-9b

| item | p(mode) | mode | top answers |
|---|---|---|---|
| s15 | 0.06 (tie) | 12745893 | 47291053×1, 48261093×1, 72041596×1, 73218495×1 |
| s16 | 0.06 (tie) | allow | kyla×1, kipo×1, tong×1, xbqh×1 |
| p6 | 0.12 (tie) | #3498db | #3498db×2, #ffd700×2, #7289da×1, #2b5797×1 |
| s2 | 0.12 | 48291 | 48291×2, 84321×1, 38294×1, 47823×1 |
| s14 | 0.12 | 847291 | 847291×2, 827451×1, 748293×1, 482193×1 |
| s17 | 0.12 | flibber | flibber×2, flibberz×1, zibblum×1, flibbert×1 |
| s21 | 0.12 | #a3f2 | #a3f2×2, #3a7f2c×1, #ffe6c7×1, #f7d2b5×1 |
| s23 | 0.12 (tie) | 7 of spades | 7♣×2, k♠×2, 7♠×2, 7 of spades×2 |
| s24 | 0.12 (tie) | 142739 | 172341×2, 142739×2, 72341×2, 172944×1 |
| s29 | 0.12 | a7k9m2 | a7k9m2×2, k7m9xp×1, x9k2m7×1, a7k9x2×1 |
| s33 | 0.12 (tie) | aethelgard | vespera×2, aethelgard×2, aetheris×2, aetherion×2 |
| r3 | 0.19 | 3847 | 3847×3, 3842×2, 7382×2, 7429×2 |
| s9 | 0.19 | flibber | flibber×3, flimzo×2, florp×2, zivlor×1 |
| s22 | 0.19 (tie) | hhhtt | httht×3, hhhtt×3, hthht×2, hhtth×2 |
| s26 | 0.19 | 03/17 | 03/17×3, 03/15×2, 06/17×2, 03/14×1 |
| s28 | 0.19 (tie) | 523 | 757×3, 523×3, 653×1, 101×1 |
| s31 | 0.19 | lumora | lumora×3, aurorn×1, auralis×1, vexȭa×1 |
| s3 | 0.25 | 1947 | 1947×4, 1974×2, 1963×2, 1973×2 |
| s4 | 0.25 | xyz | xyz×4, xab×1, tax×1, qxp×1 |
| s8 | 0.25 | apple | apple×4, tree×3, house×2, dog×2 |
| s13 | 0.25 (tie) | bubbles | orion×4, bubbles×4, goldie×3, blaze×1 |
| s19 | 0.25 | neon echo | neon echo×4, neon static×3, silent tides×1, neon ghosts×1 |
| s11 | 0.31 | helium | helium×5, gold×4, oxygen×3, carbon×2 |
| s20 | 0.31 | thorne | thorne×5, vance×3, sterling×3, varek×1 |
| s27 | 0.31 | 47.3 | 47.3×5, 42.7×2, 73.4×2, 37.4×2 |
| r2 | 0.38 | 42 | 42×6, 842957×1, 128×1, 482931×1 |
| s6 | 0.38 (tie) | 0.47 | 0.47×6, 0.73×6, 0.37×2, 0.67×1 |
| p7 | 0.44 | elara | elara×7, elian×3, lyra×2, kaelen×1 |
| s10 | 0.44 | titanic | titanic×7, the matrix×5, the sixth sense×1, the godfather part iii×1 |
| s32 | 0.44 | 3/7 | 3/7×7, 7/3×5, 7/4×3, 4/7×1 |
| s25 | 0.50 | 14:37 | 14:37×8, 14:32×3, 14:23×2, 14:27×2 |
| r4 | 0.56 | k | k×9, q×3, m×2, z×1 |
| s12 | 0.56 | texas | texas×9, california×3, florida×3, alaska×1 |
| p2 | 0.62 | guitar | guitar×10, piano×4, violin×1, guiter×1 |
| r5 | 0.69 | 4 | 4×11, 6×3, d×1, 5×1 |
| p3 | 0.69 | pepperoni | pepperoni×11, mushrooms×2, margherita×2, mushroom×1 |
| s7 | 0.69 | tokyo | tokyo×11, paris×3, london×2 |
| r1 | 0.75 | 7 | 7×12, 5×2, 4×2 |
| p5 | 0.81 | chess | chess×13, monopoly×2, catan×1 |
| s18 | 0.81 | aethelgard | aethelgard×13, eldoria×1, vylora×1, aetheris×1 |
| p1 | 0.88 | mango | mango×14, banana×2 |
| p4 | 0.94 | france | france×15, germany×1 |
| s1 | 0.94 | 42 | 42×15, 73×1 |
| s5 | 0.94 | 37 | 37×15, 57×1 |
| s30 | 0.94 | a,b | a,b×15, a,z×1 |
| d1 | 1.00 | 4 | 4×16 |
| d2 | 1.00 | paris | paris×16 |
| d3 | 1.00 | blue | blue×16 |
