# `.assets/` — README images

| File | Status | Where it appears | Notes |
|---|---|---|---|
| `namidb-logo.jpeg` | ✅ in repo | Top of `README.md`, hero block | Wordmark + isotype + tagline. Sourced from `namidb.com`. |
| `namidb-deployments.png` | ⏳ todo | "Three deployments, one engine" section | Suggested **1560×600** (2× DPI of 780×300) PNG or SVG. Three columns (Embedded / Server / Cloud) with one engine icon at the centre and arrows showing the same binary fanning out to each tier. |
| `namidb-architecture.png` | ⏳ todo | "Architecture" section | Suggested **1640×800** (2× DPI of 820×400) PNG or SVG. Parser → logical plan → optimizer → executor on top, LSM + SST + manifest CAS in the middle, S3 / R2 / GCS / Azure at the bottom, with caches as side-cars. |

All three placeholders use a `<p align="center"><img …></p>` block, so
GitHub will render them centred at the requested width. Higher-DPI
source files (2× the displayed width) keep them sharp on retina
displays.

Keep file names lowercase + kebab-case so they survive across
filesystems and CDNs.
