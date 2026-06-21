# README image assets

| File | Where it appears | Notes |
|---|---|---|
| `namidb_2.png` | Top of `README.md`, hero block | Wide "Oracle of Graphs" key art — wordmark, tagline, and the feature list. 1536×1024. |
| `namidb_3.png` | `README.md` "Architecture" section | "The bucket is the database" key art — the graph pouring into object storage. 1774×887. |
| `logo_namidb.png` | `README.md` footer; canonical brand mark | Square NamiDB logo (isotype + wordmark), for the footer, favicons, and packaging. 1254×1254. |
| `namidb-logo.jpeg` | *(superseded)* | Old wordmark strip, kept for history. Replaced as the hero by `namidb_2.png`. |
| `namidb-architecture.svg` (+ `-dark`) | *(not currently embedded)* | Layered engine diagram (parser → optimizer → executor; LSM/SST/manifest; object stores). Available for docs; light + dark variants. |
| `namidb-deployments.svg` (+ `-dark`) | *(not currently embedded)* | "Three deployments, one engine" (Embedded / Server / Cloud). Available for docs; light + dark variants. |

The PNG key art is embedded as plain centered `<img>` tags. The SVG diagrams
ship light and dark variants (`*-dark.svg`) for a future `<picture>` block, but
are not referenced from `README.md` today.

Keep file names lowercase so they survive across filesystems and CDNs.
