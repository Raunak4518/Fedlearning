"""E02 - Does weight decay collapse rare-class conditioning?  (plan 2.1, E1.3, E4.9)

Derivation (plan eq. 2.2): with Adam + coupled L2 (CVAE-F, Table XV), a
client with no class-r sample moves row w_r toward 0 at ~lr per step, and
flat averaging lets non-holders outvote holders: |w_r| cannot grow when
fewer than half the clients hold class r. Predictions:
 (i)   under GeFL-F, tail-row norms shrink relative to head rows;
 (ii)  HWA (holders-only weighting), LCD (decay only present rows) or no
       weight decay each stop the collapse;
 (iii) tail generator fidelity (a frozen referee classifier's accuracy on
       synthetic tail features) rises when the collapse is removed.
Logged per run: cond_norm_tail_over_head_end, cond_norm_tail_rel_init,
fidelity_head / fidelity_tail, and downstream accuracy.
"""
GENS = [("GeFL-F", ""), ("+NOWD", "NOWD"), ("+HWA", "HWA"), ("+LCD", "LCD"),
        ("+HWA+LCD", "HWA+LCD"), ("+NOWD+HWA", "NOWD+HWA")]
EXPERIMENT = dict(
    name="E02_conditioning_collapse",
    title="Rare-class conditioning collapse under weight decay and flat averaging",
    hypothesis="GeFL-F tail rows collapse (norm ratio << 1, low tail fidelity); HWA / LCD / NOWD prevent it.",
    reference="GeFL-F",
    metrics=["cond_norm_tail_over_head_end", "cond_norm_tail_rel_init", "fidelity_tail", "fidelity_head",
             "final_bal", "final_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"] for lab, c in GENS],
)
