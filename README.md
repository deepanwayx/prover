# Prover

The repository contains the code for the paper [Prover: Generating Intermediate Steps for NLI with Commonsense Knowledge Retrieval and Next-Step Prediction]() published at IJCNLP-AACL 2023.

## Abstract

The Natural Language Inference (NLI) task often requires reasoning over multiple steps to reach the conclusion. While the necessity of generating such intermediate steps (instead of a summary explanation) has gained popular support, *it is unclear how to generate such steps without complete end-to-end supervision and how such generated steps can be further utilized*. In this work, we train and enhance a sequence-to-sequence next-step prediction model with external commonsense knowledge and search to generate intermediate steps with limited next-step supervision. We show the correctness of such generated steps through human verification, on MNLI and MED datasets (and discuss the limitations through qualitative examples). We show that such generated steps can help improve end-to-end NLI task performance using simple data augmentation strategies. Using a CheckList dataset for NLI, we also explore the effect of augmentation on specific reasoning types.


## Contents

1. Training the next-step generation and sentence composition models: [training](https://github.com/deepanwayx/prover/tree/main/training/README.md)
2. Proof generation: [proof_generation](https://github.com/deepanwayx/prover/tree/main/proof_generation/README.md)
3. NLI Classification experiments with augmented data: [classification](https://github.com/deepanwayx/prover/tree/main/classification/README.md)
4. Human verified proofs: [proofs](https://github.com/deepanwayx/prover/blob/main/proofs/prover_psf.csv)


## Citation
Please consider citing the following article if you found our work useful:

```bibtex
@inproceedings{ghosal2023prover,
  title={Prover: Generating Intermediate Steps for NLI with Commonsense Knowledge Retrieval and Next-Step Prediction},
  author={Ghosal, Deepanway and Aditya, Somak and Choudhury, Monojit},
  booktitle={Proceedings of the 2023 International Joint Conference on Natural Language Processing and the Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  year={2023}
}
```