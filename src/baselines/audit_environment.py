import importlib, json, platform, shutil, sys

mods = {}
for name in ("numpy", "pandas", "sklearn", "torch", "xgboost"):
    try:
        mod = importlib.import_module(name)
        mods[name] = {"ready": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:
        mods[name] = {"ready": False, "error": type(exc).__name__}

report = {
    "python": sys.version.split()[0],
    "architecture": platform.machine(),
    "modules": mods,
    "Rscript": shutil.which("Rscript"),
    "gzip": shutil.which("gzip"),
}
print(json.dumps(report, indent=2))

