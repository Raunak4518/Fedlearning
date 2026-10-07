"""E07 - Ablations: justify every number in the method.  (plan P4)

MNIST, IF=100 + Dir(0.5), 2 seeds (screening). Varied one at a time
around Ours: LA temperature tau; HWA weight form (E(n) at several beta,
linear n, and the opposite direction 1/E(n)); PCM mixing ratio; the
sequential synthetic-phase variants PCM replaces (gap-filling with local vs
global counts, real-first order, a larger synthetic share T_s:T_r); BCR steps.
"""


def _v(lab, parts, **over):
    return (lab, compose(parts, **over))


VARIANTS = [
    _v("Ours (HWA+LCD+PCM+LA)", OURS),
    _v("tau=0.5", OURS, head=dict(tau=0.5)), _v("tau=1.5", OURS, head=dict(tau=1.5)),
    _v("tau=2.0", OURS, head=dict(tau=2.0)),
    _v("HWA beta=0.9", OURS, gen=dict(beta=0.9)), _v("HWA beta=0.9999", OURS, gen=dict(beta=0.9999)),
    _v("HWA linear n", OURS, gen=dict(weight="linear")),
    _v("HWA inverse 1/E(n)", OURS, gen=dict(weight="inverse")),
    _v("PCM r=0.1", OURS, head=dict(mix_ratio=0.1)), _v("PCM r=0.5", OURS, head=dict(mix_ratio=0.5)),
    _v("PCM r=1.0", OURS, head=dict(mix_ratio=1.0)),
    _v("seq GF local (HWA+LCD+GF+LA)", "HWA+LCD+GF+LA"),
    _v("seq GF global", "HWA+LCD+LA", head=dict(sampler="gapfill_global")),
    _v("seq uniform real-first", "HWA+LCD+LA+RF"),
    _v("seq GF Ts:Tr=3:3", "HWA+LCD+GF+LA", head=dict(ts=3, tr=3)),
    _v("BCR 10 steps", OURS, head=dict(bcr=10)), _v("BCR 100 steps", OURS, head=dict(bcr=100)),
]
EXPERIMENT = dict(
    name="E07_ablations",
    title="Ablations of the method's design choices",
    hypothesis="tau near 1 is best; E(n) ~ linear n >> inverse; PCM beats every sequential variant.",
    reference="Ours (HWA+LCD+PCM+LA)",
    seeds=[0, 1],
    plan=lambda cfg: [run_spec(lab, "mnist", s, m, IF=0.01, alpha=0.5) for s in cfg["seeds"] for lab, m in VARIANTS],
)
