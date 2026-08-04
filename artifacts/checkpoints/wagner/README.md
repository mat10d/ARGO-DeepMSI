# Wagner MSI checkpoint

Place the published `MSI_high_CRC_model.pth` checkpoint in this directory. The binary is
gitignored; `manifest.json` records its upstream provenance and required SHA-256 digest.

Runtime code resolves this location by default and refuses a checksum mismatch. An alternate
location may be supplied through `ARGO_WAGNER_CHECKPOINT`.

The archived `old/HistoBistro` checkout is not a runtime dependency.
