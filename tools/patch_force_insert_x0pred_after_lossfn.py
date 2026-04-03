import re, sys, shutil
from pathlib import Path

path = Path("mpd/models/diffusion_models/diffusion_model_base.py")
txt = path.read_text(encoding="utf-8", errors="ignore")
bak = path.with_suffix(path.suffix + ".bak_force_insert2")
shutil.copy2(path, bak)
print("[OK] backup ->", bak)

# Find p_losses block
m = re.search(r"\n(\s*)def p_losses\(self, x_start, context_d, t, hard_conds\):\n", txt)
if not m:
    print("[ERR] cannot find p_losses signature")
    sys.exit(1)

indent = m.group(1)
start = m.end()
m2 = re.search(r"\n" + re.escape(indent) + r"def ", txt[start:])
end = len(txt) if not m2 else start + m2.start()
pl = txt[m.start():end]

# Avoid double insert
if "info[\"x_recon\"]" in pl or "info['x_recon']" in pl:
    print("[OK] already has x_recon in p_losses, no change.")
    sys.exit(0)

lines = pl.splitlines(True)
out = []
inserted = False

# We insert right AFTER the loss_fn assignment (either branch), before return.
for i, line in enumerate(lines):
    out.append(line)

    # match either "loss, info = self.loss_fn(eps_pred, noise)" or "(..., x_start)"
    if (not inserted) and re.search(r"^\s*loss\s*,\s*info\s*=\s*self\.loss_fn\(", line):
        ind = re.match(r"^(\s*)loss", line).group(1)
        inject = (
            f"{ind}# --- expose predicted x0 for planner-like losses ---\n"
            f"{ind}if not isinstance(info, dict):\n"
            f"{ind}    info = {{}}\n"
            f"{ind}if self.predict_epsilon:\n"
            f"{ind}    x0_pred = self.predict_start_from_noise(x_noisy, t=t, noise=eps_pred)\n"
            f"{ind}else:\n"
            f"{ind}    x0_pred = eps_pred\n"
            f"{ind}x0_pred = apply_hard_conditioning(x0_pred, hard_conds)\n"
            f"{ind}info[\"x_recon\"] = x0_pred\n"
        )
        out.append(inject)
        inserted = True

if not inserted:
    print("[ERR] could not find loss_fn assignment to insert after")
    sys.exit(1)

new_pl = "".join(out)
new_txt = txt[:m.start()] + new_pl + txt[end:]
path.write_text(new_txt, encoding="utf-8")
print("[OK] inserted info['x_recon'] into p_losses")