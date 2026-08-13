# V0 result — one-seed signal run

Run: `signal-20260812-131246`  
Model: `Qwen/Qwen3-1.7B`  
Method: rank-64 LoRA, seed 11  
GPU: one H100-SXM

## Bottom line

Tiny constitution-specific SDF can move held-out decision probabilities in
this setup, but V0 did not yet demonstrate broad specification
internalization. The clean effect was a C2 shift on authority-versus-ownership
cases. The other two disagreement motifs were null or moved in the wrong
direction, and C1 did not improve over a control already biased toward C1.

## Actual dose

- C1 SDF: 6,851 unique tokens, 27,404 processed tokens, 7 optimizer updates.
- C2 SDF: 6,719 unique tokens, 26,876 processed tokens, 7 optimizer updates.
- Identical SFT: 192 examples, 8 epochs, 96 optimizer updates in every arm.

## Main measurements

- Balanced explicit-disagreement probability separation: **+0.0571**.
- Log-odds separation: **+0.0338**.
- C1 preferred-probability uplift versus control: **-0.0308**.
- C2 preferred-probability uplift versus control: **+0.0900**.
- Agreement accuracy: control 96%, C1 96%, C2 95%.
- Near-miss accuracy: control 97%, C1 98%, C2 93%.

Separation by motif:

- Authority vs ownership: **+0.2389**.
- Authority vs stewardship: **-0.0475**.
- Effectiveness vs reversibility: **-0.0201**.

The no-SDF control already preferred C1 on explicit disagreement cases:
68.3% C1 accuracy versus 24.0% C2 accuracy. C2 SDF changed this to 58.0% C1
versus 33.3% C2 while keeping agreement accuracy at 95%. C1 SDF changed it to
65.3% C1 versus 28.0% C2.

## Capability warning

With the relevant constitution supplied in context, the untouched model scored
only 36% on C1 and 39% on C2 (25% chance). This means a broad behavioral null
cannot be attributed to SDF scale: the present 1.7B model cannot reliably
reconstruct the numerical oracle from the prose constitution.

## Analysis correction

The first logged aggregate was +0.103 because it mixed the balanced 300-item
disagreement bank with an implicit subset that was accidentally 99%
authority-versus-ownership. The corrected headline uses only the balanced
explicit bank. Future implicit generation is now balanced 34/33/33 across the
three motifs.

## Next smallest useful experiment

1. Make the constitution-to-oracle mapping easier for the same model, or test a
   larger checkpoint in context, until the capability diagnostic clearly shows
   that the eval is solvable.
2. Add a small direct constitution-knowledge diagnostic and a direct
   disagreement-SFT positive control.
3. With that substrate frozen, measure C1/C2 separation at roughly
   4k, 16k, and 64k unique SDF tokens, one seed for screening and more seeds
   only for a promising point.

The first run therefore answers the narrow scale question positively: a
measurable probability effect is visible below 8k unique tokens. It does not
yet establish that the effect reflects the full intended latent value ordering.
