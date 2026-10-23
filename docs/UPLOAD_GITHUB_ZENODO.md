# GitHub and Zenodo release checklist

## GitHub

1. Create an empty public repository named `TRACE-GS`; do not initialize it with another README or license.
2. Upload the **contents** of this directory, not the enclosing directory or the ZIP file alone.
3. Verify that `README.md`, `LICENSE`, `CITATION.cff`, `VERSION`, `MANIFEST.sha256`, `src/`, `splits/`, and `results/` appear at the repository root.
4. Commit with message `TRACE-GS v1.0.0 reproducibility release`.
5. Create tag and GitHub Release `v1.0.0`; attach `TRACE-GS-release-v1.0.0.zip` as a release asset.
6. Add topics such as `genomic-selection`, `multi-omics`, `transcriptomics`, and `plant-breeding`.

## Zenodo

1. Sign in to Zenodo with GitHub and enable the `TRACE-GS` repository.
2. Create a new GitHub Release only after the Zenodo connection is enabled, or reissue the release if Zenodo does not detect it.
3. Before publishing the Zenodo record, complete the author list, affiliations, ORCIDs, related manuscript information, keywords, and MIT licence metadata.
4. Publish the Zenodo record and copy both the version DOI and concept DOI into the manuscript and `CITATION.cff`.

## Required checks before manuscript submission

```bash
python src/data_audit/validate_release.py
python tests/run_smoke_tests.py
```

Compare the uploaded release asset against the locally recorded SHA-256 value. Do not upload raw third-party genotype, transcriptome or phenotype files.

