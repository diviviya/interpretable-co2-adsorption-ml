"""
apply_selection_patch.py: adds the "Find materials" feature to your 3 existing files.

Run it ONCE from the repo root (the folder that contains backend/ and frontend/):

    python apply_selection_patch.py

It changes:
  backend/app/main.py      -> registers the new selection router
  frontend/js/api.js       -> adds the API.selection... calls
  frontend/js/common.js    -> adds "Find materials" to the top menu

It is safe to run twice (it skips anything already done) and it stops with a
clear message if a file doesn't look the way it expects, without touching it.
Use `git diff` afterwards to see exactly what changed.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def patch(rel_path, edits):
    path = ROOT / rel_path
    if not path.exists():
        sys.exit(f"STOP: {rel_path} not found. Run this from the repo root.")
    raw = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    changed = False
    for name, already, apply in edits:
        if already in text:
            print(f"  skip   {rel_path}: {name} (already done)")
            continue
        new = apply(text)
        if new is None or new == text:
            sys.exit(f"STOP: couldn't find the place to add '{name}' in {rel_path}. Nothing was changed in that file.")
        text = new
        changed = True
        print(f"  done   {rel_path}: {name}")
    if changed:
        path.write_bytes(text.replace("\n", nl).encode("utf-8"))


def replace_once(old, new):
    def go(text):
        return text.replace(old, new, 1) if old in text else None
    return go


# ---------------- backend/app/main.py ----------------
patch("backend/app/main.py", [
    ("import selection router",
     "predict, selection",
     replace_once("from .routers import admin, auth, health, predict\n",
                  "from .routers import admin, auth, health, predict, selection\n")),
    ("include selection router",
     "app.include_router(selection.router)",
     replace_once("app.include_router(predict.router)\n",
                  "app.include_router(predict.router)\napp.include_router(selection.router)\n")),
])

# ---------------- frontend/js/api.js ----------------
patch("frontend/js/api.js", [
    ("demo-mode helper",
     "const demoOff",
     replace_once("  const m = () => CONFIG.USE_MOCK;\n",
                  "  const m = () => CONFIG.USE_MOCK;\n"
                  "  // The material finder needs the real model, so it is not available in demo mode.\n"
                  "  const demoOff = () => Promise.reject(new Error(\"The material finder needs the real backend (turn off demo mode).\"));\n")),
    ("selection calls",
     "selectionScreen",
     replace_once("    // ---- admins (the backend checks the role on every call) ----\n",
                  "    // ---- material finder (logged-in users) ----\n"
                  "    selectionCarbons: () => m() ? demoOff() : request(\"/api/selection/carbons\"),\n"
                  "    selectionScreen:  (body) => m() ? demoOff() : request(\"/api/selection/screen\",  { method: \"POST\", body }),\n"
                  "    selectionWhatIf:  (body) => m() ? demoOff() : request(\"/api/selection/what-if\", { method: \"POST\", body }),\n"
                  "    selectionSweep:   (body) => m() ? demoOff() : request(\"/api/selection/sweep\",   { method: \"POST\", body }),\n\n"
                  "    // ---- admins (the backend checks the role on every call) ----\n")),
])


# ---------------- frontend/js/common.js ----------------
def add_nav(text):
    m = re.search(r'^([ \t]*)\{ href: "batch\.html".*$', text, flags=re.M)
    if not m:
        return None
    line = f'{m.group(1)}{{ href: "selector.html", label: "Find materials", page: "selector", needs: "user" }},'
    return text[:m.start()] + line + "\n" + text[m.start():]


patch("frontend/js/common.js", [
    ("menu link", 'href: "selector.html"', add_nav),
])

print("\nAll done. Restart the backend (uvicorn) and open the 'Find materials' page.")
