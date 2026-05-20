# README image assets

| File | Status | Where it appears | Notes |
|---|---|---|---|
| `namidb-logo.jpeg` | in repo | Top of `README.md`, hero block | Wordmark + isotype + tagline. Sourced from namidb.com. |
| `namidb-deployments.svg` (+ `-dark`) | in repo | "Three deployments, one engine" | Three columns (Embedded / Server / Cloud), one engine icon in the middle, arrows showing the same binary fanning out to each tier. Light and dark variants. |
| `namidb-architecture.svg` (+ `-dark`) | in repo | "Architecture" | Parser -> logical plan -> optimizer -> executor on top, LSM + SST + manifest CAS in the middle, S3 / R2 / GCS / Azure at the bottom, caches off to the side. Light and dark variants. |

The two diagrams are pulled into `README.md` through a `<picture>` block
so GitHub serves the `-dark` variant when the reader is in dark mode.
The logo is a plain centered `<img>`.

Keep file names lowercase and kebab-case so they survive across
filesystems and CDNs.
