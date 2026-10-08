"""F06 - Mechanism (full rounds, 3 seeds): why GeFL-F's generator fails on rare classes.

Full-schedule version of E02. Under the paper's CVAE-F weight decay and
flat averaging, rare-class conditioning rows collapse (plan eq. 2.2) and
rare-class generator fidelity falls; HWA, LCD or no decay stop the
collapse; FSG side-steps it (exact aggregation). Each with the paper's
stage (iii), so accuracy differences come from the generator alone.
"""
EXPERIMENT = dict(
    name="F06_mechanism",
    title="Generator mechanism: conditioning collapse and its fixes",
    hypothesis="GeFL-F tail-row norm ratio << 1 and low tail fidelity; HWA/LCD/NOWD restore norms; FSG has the "
               "highest tail fidelity.",
    reference="GeFL-F",
    metrics=["cond_norm_tail_over_head_end", "cond_norm_tail_rel_init", "fidelity_tail", "fidelity_head",
             "final_bal", "final_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"]
                      for lab, c in [("GeFL-F", ""), ("+NOWD", "NOWD"), ("+HWA", "HWA"), ("+LCD", "LCD"),
                                     ("FSG", "GAUSS")]],
)
