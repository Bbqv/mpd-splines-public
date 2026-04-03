import re, sys
from pathlib import Path

path = Path("mpd/models/diffusion_models/diffusion_model_base.py")
txt = path.read_text(encoding="utf-8", errors="ignore")

# Find p_losses definition block
m = re.search(r"\n(\s*)def p_losses\(self, x_start, context_d, t, hard_conds\):\n", txt)
if not m:
    print("[ERR] cannot find p_losses signature exactly. Please open file and check signature.")
    sys.exit(1)

indent = m.group(1)
start = m.end()

# Find end of p_losses by next def at same indent
m2 = re.search(r"\n" + re.escape(indent) + r"def ", txt[start:])
end = len(txt) if not m2 else start + m2.start()
pl = txt[m.start():end]

# Sanity: ensure we have the expected lines
if "x_recon = self.model" not in pl:
    print("[ERR] p_losses doesn't contain `x_recon = self.model` as expected.")
    sys.exit(1)
if "return loss, info" not in pl:
    print("[ERR] p_losses doesn't end with `return loss, info` as expected.")
    sys.exit(1)

# Patch: rename x_recon to eps_pred and inject x0_pred into info
pl_new = pl

# 1) rename the model output variable (only first two assignments)
pl_new = pl_new.replace("x_recon = self.model(x_noisy, t, context_emb)", "eps_pred = self.model(x_noisy, t, context_emb)", 1)
pl_new = pl_new.replace("x_recon = apply_hard_conditioning(x_recon, hard_conds)", "eps_pred = apply_hard_conditioning(eps_pred, hard_conds)", 1)

# 2) update assert noise.shape == x_recon.shape
pl_new = pl_new.replace("assert noise.shape == x_recon.shape", "assert noise.shape == eps_pred.shape", 1)

# 3) update loss_fn calls
pl_new = pl_new.replace("loss, info = self.loss_fn(x_recon, noise)", "loss, info = self.loss_fn(eps_pred, noise)", 1)
pl_new = pl_new.replace("loss, info = self.loss_fn(x_recon, x_start)", "loss, info = self.loss_fn(eps_pred, x_start)", 1)

# 4) inject x0_pred into info for predict_epsilon path (and also for non-epsilon path, treat eps_pred as x_start_pred already)
inject = f"""
{indent}    # --- expose predicted x0 for external planner-like losses ---
{indent}    if not isinstance(info, dict):
{indent}        info = {{}}
{indent}    if self.predict_epsilon:
{indent}        # eps_pred -> x0_pred
{indent}        x0_pred = self.predict_start_from_noise(x_noisy, t=t, noise=eps_pred)
{indent}        x0_pred = apply_hard_conditioning(x0_pred, hard_conds)
{indent}        info["x_recon"] = x0_pred
{indent}    else:
{indent}        # model predicts x0 directly
{indent}        x0_pred = eps_pred
{indent}        x0_pred = apply_hard_conditioning(x0_pred, hard_conds)
{indent}        info["x_recon"] = x0_pred
"""

# Insert injection right before `return loss, info`
pl_new = pl_new.replace(f"\n{indent}    return loss, info\n", inject + f"\n{indent}    return loss, info\n", 1)

if pl_new == pl:
    print("[ERR] patch made no changes. Aborting.")
    sys.exit(1)

new_txt = txt[:m.start()] + pl_new + txt[end:]
path.write_text(new_txt, encoding="utf-8")
print("[OK] patched p_losses: now returns info['x_recon']=x0_pred")