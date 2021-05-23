import importlib
from pathlib import Path

my_dir = Path(__file__).parent

page_to_module = {}

for f in my_dir.iterdir():
    if f.name.startswith('_'):
        continue

    if f.suffix == '.py':
        modname = my_dir.name + '.' + f.stem
        mod = importlib.import_module(modname)

        if hasattr(mod, 'layout') and hasattr(mod, 'PAGENAME'):
            page_to_module[getattr(mod, 'PAGENAME')] = mod

del my_dir, f, modname, mod, importlib, Path
