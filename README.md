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