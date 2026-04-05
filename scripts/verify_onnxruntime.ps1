# Quick ORT sanity check on Windows (DLL / VC++ redistributable issues).
$ErrorActionPreference = "Stop"
try {
  python -c "import onnxruntime as ort; print('ORT', ort.__version__, 'OK')"
} catch {
  Write-Host "ONNXRuntime failed to import. Try: pip install --force-reinstall onnxruntime"
  Write-Host "Also install the latest Microsoft Visual C++ Redistributable (x64)."
  exit 1
}
