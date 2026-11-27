# GFP distance-pair ablation

This isolated benchmark branch is limited to a single full-length Aequorea victoria GFP-family protein (sequence reference [UniProt P42212](https://www.uniprot.org/uniprot/P42212)). The experiment compares zero added pairs with nested 4, 8, 12 and 16-pair subsets of the existing surface-elevation guide. The full guide is generated and checked before any subset is removed. All arms use the same prepared scaffold and matched replicate seeds. No new angle definition, inverse-folding settings or refold settings are introduced.

`--gfp-pair-ablation` is an explicit option to the existing surface-elevation launcher. Its `--backbones` argument retains its usual per-arm meaning: 200 per arm gives 1000 total. The report and CSVs record encoded and informative pair counts separately. This is a GFP geometry benchmark, not a production pipeline change. The initial sample size is suited to describing geometric compliance, not to establishing differences in rare recovery outcomes.

