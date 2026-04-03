import io, re, sys, shutil
from pathlib import Path

path = Path("mpd/models/diffusion_models/diffusion_model_base.py")
txt = path.read_text(encoding="utf-8", errors="ignore")

bak = path.with_suffix(path.suffix + ".bak_xrecon")
shutil.copy2(path, bak)
print("[OK] backup ->", bak)

# Find first occurrence of x_recon assignment inside loss()
m = re.search(r"\n(\s*)x_recon\s*=\s*self\.predict_start_from_noise\(", txt)
if not m:
    # fallback: maybe named x_start / x0
    m = re.search(r"\n(\s*)x_recon\s*=\s*self\.predict_start_from_noise", txt)
if not m:
    print("[ERR] cannot find `x_recon = self.predict_start_from_noise(` in file.")
    sys.exit(1)

indent = m.group(1)

insertion = (
    f"\n{indent}# --- expose x_recon for external losses (planner-like) ---\n"
    f"{indent}try:\n"
    f"{indent}    info\n"
    f"{indent}except NameError:\n"
    f"{indent}    info = {{}}\n"
    f"{indent}if not isinstance(info, dict):\n"
    f"{indent}    info = {{}}\n"
    f"{indent}info[\"x_recon\"] = x_recon\n"
)

# Insert after the line that assigns x_recon ... we insert after the first newline following that assignment line block.
# Safer: insert right after the first occurrence of that assignment line (the 'x_recon = ...' line).
lines = txt.splitlines(True)
out = []
inserted = False
for i, line in enumerate(lines):
    out.append(line)
    if (not inserted) and re.search(rf"^\s*x_recon\s*=\s*self\.predict_start_from_noise", line):
        out.append(insertion)
        inserted = True

if not inserted:
    print("[ERR] found regex but failed to insert.")
    sys.exit(1)

new_txt = "".join(out)
path.write_text(new_txt, encoding="utf-8")
print("[OK] patched", path)