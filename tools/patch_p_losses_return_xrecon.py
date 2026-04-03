import re, sys, shutil
from pathlib import Path

path = Path("mpd/models/diffusion_models/diffusion_model_base.py")
txt = path.read_text(encoding="utf-8", errors="ignore")

bak = path.with_suffix(path.suffix + ".bak_p_losses_xrecon")
shutil.copy2(path, bak)
print("[OK] backup ->", bak)

# 1) locate p_losses block
m = re.search(r"\n(\s*)def p_losses\(", txt)
if not m:
    print("[ERR] cannot find `def p_losses(`")
    sys.exit(1)

pl_start = m.start()
indent_def = m.group(1)

# Find end of p_losses by next "def " at same indent
pattern_next_def = r"\n" + re.escape(indent_def) + r"def "
m2 = re.search(pattern_next_def, txt[m.end():])
pl_end = len(txt) if not m2 else (m.end() + m2.start())

pl = txt[pl_start:pl_end]

# 2) Ensure x_recon exists in p_losses
if not re.search(r"^\s*x_recon\s*=\s*self\.predict_start_from_noise", pl, flags=re.M):
    # maybe named x_start
    if re.search(r"predict_start_from_noise", pl) is None:
        print("[ERR] p_losses does not contain predict_start_from_noise; need manual inspection.")
        sys.exit(1)
    print("[ERR] found predict_start_from_noise but not assigned to x_recon with that name.")
    print("      Please inspect p_losses and adjust this patch script accordingly.")
    sys.exit(1)

# 3) Patch return inside p_losses:
# Handle common patterns:
#   a) return loss, info
#   b) return self.loss_fn(pred, targ)   (which returns (loss, {}))
#   c) loss, info = self.loss_fn(...); return loss, info
#
# We'll patch the "return ..." line that returns the loss in p_losses.
lines = pl.splitlines(True)

out = []
patched = False
for i, line in enumerate(lines):
    # case c: `return loss, info`
    if (not patched) and re.match(r"^\s*return\s+loss\s*,\s*info\s*$", line.strip()):
        ind = re.match(r"^(\s*)return", line).group(1)
        out.append(f"{ind}info = info if isinstance(info, dict) else {{}}\n")
        out.append(f"{ind}info['x_recon'] = x_recon\n")
        out.append(f"{ind}return loss, info\n")
        patched = True
        continue

    # case b: return self.loss_fn(...)
    if (not patched) and re.search(r"^\s*return\s+self\.loss_fn\(", line):
        ind = re.match(r"^(\s*)return", line).group(1)
        # replace with explicit unpack + info inject + return
        out.append(f"{ind}loss, info = self.loss_fn(pred, targ)\n")
        out.append(f"{ind}info = info if isinstance(info, dict) else {{}}\n")
        out.append(f"{ind}info['x_recon'] = x_recon\n")
        out.append(f"{ind}return loss, info\n")
        patched = True
        continue

    # case a: generic `return <something>` at end: we don't touch, unless it returns tuple from loss_fn
    out.append(line)

if not patched:
    # fallback: look for `loss, info = self.loss_fn(` then `return loss, info` separately
    # If neither matched, we cannot safely patch.
    print("[ERR] could not find a return pattern to patch inside p_losses.")
    print("      Please open p_losses section and check how it returns loss.")
    sys.exit(1)

new_pl = "".join(out)
new_txt = txt[:pl_start] + new_pl + txt[pl_end:]
path.write_text(new_txt, encoding="utf-8")
print("[OK] patched p_losses to return info['x_recon']")