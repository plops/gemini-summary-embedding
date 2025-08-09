# Contents

|---------------+----------------------------------------------+---------------------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------|
| file          | in                                           | cache                     | out                                                        | purpose                                                                                                                           |
|---------------+----------------------------------------------+---------------------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------|
| 01/embed.py   | sqlite                                       |                           | sqlite                                                     | call google gemini to compute embeddings for each video summary (cost .15USD/1Mtoken, free tier limited to 1000 requests per day) |
| 01/cluster.py | sqlite, reducer.pkl, parts.csv, fulltext.csv |                           |                                                            | runs DBSCAN to find clusters                                                                                                      |
| 01/loaddb.py  | sqlite                                       | reducer.pkl               | reducer.pkl, parts.csv, fulltext.csv, embeddings.npy, html | load the sqlite3 database with embeddings and use umap to visualize                                                               |
| 02/main.py    | embeddings.npy, parts.csv, fulltext.csv      | reducer.pkl, clusters.pkl | main.html                                                  | visualize 2D UMAP embeddings, with cluster colors and names from DBSCAN clustering on 4D UMAP                                     |

# Dependencies

Check if a python is already present
```
kiel@localhost ~/stage/embed $ uv python list
cpython-3.13.5-linux-x86_64-gnu    /usr/bin/python3.13
```

Yes, we have a python. if not use `uv python install`.

As described here:
https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies

a TOML comment at the top of the source file declares its Python dependencies.

Numba requires python 3.12, download like so: `uv python install 3.12`.
`uv sync` downloads all dependencies

Run individual scripts:
```
uv run embed.py
uv run cluster.py
uv run loaddb.py
```

Run tests:
```
uv run pytest
```


Use `uv` to install them like by running like this:
```
uv run embed.py
```

Format python files
```
uv run ruff format
```

Check python code (detects unused imports). Use --fix to try automatic correction.
```
uv run ruff check
```

Using UV in VSCode. Install Ruff extension
Execute `C-S-p Restart Ruff` in Vscode
More information: https://github.com/astral-sh/ruff-vscode/blob/main/README.md

I created pyproject.toml and executed `uv sync`
