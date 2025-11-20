# RLBot v5 Element

This project is a fork of the RLBot v4 version of Element.

It's aim is to port the bot to RLBot v5 and to use a shared PyTorch dependency so the resulting botpack exe is smaller.

## Prerequisites

Install the normal bot requirements via `pip install -r requirements.txt`.

## Configuring for the v5 botpack

1. Importing the shared botpack PyTorch
    - PyTorch is only imported in `src/agent.py`, making this easy.
        If you `import torch` in many files, just replace every instance with the below snippet -
        unless you know which import is guaranteed to be ran first.
    - `import torch` was replaced with the following:

        ```python
        try:
            import torch
        except ImportError:
            import sys

            sys.path.insert(0, "../../torch-archive")
            import torch
        ```

    - If PyTorch is installed in the local environment, that version will be loaded.
        If a version in the local environment isn't installed, then it will try to load the botpack PyTorch redistributable.
        This assumes that the bot is already inside the botpack.
    - Packages included in the redistributable include PyTorch's dependencies downloaded from pip except for numpy, which are:
        1. filelock
        1. fsspec
        1. jinja
        1. markupsafe
        1. mpmath
        1. networkx
        1. sympy
        1. torch
        1. torchgen
        1. typing_extensions

1. `pip install pyinstaller`
1. `pyinstaller --onefile src/bot.py --paths=src --add-data=src/model.p:. --exclude-module=torch --hidden-import=timeit --hidden-import=pickletools --hidden-import=uuid --hidden-import=unittest.mock`
    - This will create a file called `bot.spec` - you may have to remove `*.spec` from `.gitignore` to commit it.
    - If your model is not called `model.p` and inside the `src` folder, you must change the path after `--add-data`.
        HOWEVER, be sure to keep `:.` at the end! It is required for PyInstaller.
    - When PyTorch loads, it also loads the std modules `timeit`, `pickletools`, `uuid`, and `unittest`.
    To prevent PyInstaller from trimming these std modules, they must be listed as hidden imports.
1. Create `bob.toml` in the same directory as the spec file with the following content:

    ```toml
    [[config]]
    project_name = "PythonExample"
    bot_configs = ["src/bot.toml"]

    [config.builder_config]
    builder_type = "pyinstaller"
    entry_file = "bot.spec"
    ```

    - `project_name` will be the name of your bot's folder in the botpack
    - `bot_configs` is a list of bot configs that will be included in the botpack
    - `builder_type` should always be `pyinstaller`
    - `entry_file` is the name of the spec file

1. Commit both `bot.spec` and `bob.toml` to your bot's repository.
    Note that `bob.toml` CANNOT be renamed, but `bot.spec` can be anything as long as `entry_file` is also renamed to reflect the change.
