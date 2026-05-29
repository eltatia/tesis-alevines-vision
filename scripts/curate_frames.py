"""
curate_frames.py
----------------
Genera un visor HTML interactivo para curar (keep/discard) los frames
extraídos de los videos. Crea thumbnails y un index.html que se abre
en el navegador y permite descartar visualmente los frames borrosos,
sin alevines, o defectuosos.

Flujo:
    1. python scripts/curate_frames.py
       -> genera reports/preview/thumbs/ y reports/preview/index.html
    2. Abrir reports/preview/index.html en el navegador
    3. Click en miniatura = marcar para DESCARTAR (overlay rojo)
       Click otra vez = quitar marca
       Atajos: "D" = descartar el actual, "K" = keep, flechas para navegar
    4. Botón "Descargar selección" guarda selection.json
    5. python scripts/apply_curation.py reports/preview/selection.json
       -> copia los frames NO descartados a data/frames/selected/

Uso:
    python scripts/curate_frames.py
    python scripts/curate_frames.py --src data/frames/all --thumb-size 320
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera visor HTML para curar frames.")
    p.add_argument("--src", type=str, default="data/frames/all",
                   help="Carpeta con los frames a curar")
    p.add_argument("--output-dir", type=str, default="reports/preview",
                   help="Carpeta donde generar thumbnails y el index.html")
    p.add_argument("--thumb-size", type=int, default=320,
                   help="Tamaño máximo del thumbnail en px (default: 320)")
    p.add_argument("--columns", type=int, default=5,
                   help="Columnas en el grid HTML (default: 5)")
    return p.parse_args()


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Curaduría de frames - alevines</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
      margin: 0; padding: 0; background: #1e1e1e; color: #e4e4e4;
    }
    header {
      position: sticky; top: 0; z-index: 10;
      background: #2d2d2d; padding: 14px 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    }
    header h1 { font-size: 18px; margin: 0; }
    .stats { font-family: monospace; padding: 6px 12px; background: #1e1e1e; border-radius: 6px; }
    .stats strong { color: #4fc3f7; }
    .stats .discarded { color: #ff5252; }
    button {
      background: #4fc3f7; border: none; color: #000;
      padding: 8px 14px; border-radius: 6px; font-weight: 600;
      cursor: pointer; font-size: 14px;
    }
    button:hover { background: #29b6f6; }
    button.danger { background: #ff5252; color: #fff; }
    button.danger:hover { background: #e64141; }
    button.ghost { background: transparent; color: #e4e4e4; border: 1px solid #555; }
    button.ghost:hover { background: #383838; }

    .grid {
      display: grid;
      grid-template-columns: repeat(COLUMNS, 1fr);
      gap: 8px; padding: 16px;
    }
    .thumb {
      position: relative; cursor: pointer;
      border-radius: 6px; overflow: hidden;
      background: #000;
      transition: transform 0.05s;
    }
    .thumb:hover { transform: scale(1.02); }
    .thumb img { display: block; width: 100%; height: auto; }
    .thumb .name {
      position: absolute; bottom: 0; left: 0; right: 0;
      background: rgba(0,0,0,0.75); padding: 4px 6px;
      font-size: 10px; font-family: monospace;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .thumb.discarded {
      opacity: 0.4;
    }
    .thumb.discarded::after {
      content: "DESCARTAR";
      position: absolute; top: 50%; left: 50%;
      transform: translate(-50%, -50%) rotate(-15deg);
      background: rgba(255,82,82,0.92); color: #fff;
      padding: 4px 12px; font-weight: 700; font-size: 13px;
      border-radius: 4px; pointer-events: none;
      letter-spacing: 1px;
    }
    .thumb.active { outline: 3px solid #4fc3f7; }
    .help {
      padding: 12px 20px; background: #252525; font-size: 13px;
      border-bottom: 1px solid #333;
    }
    .help kbd {
      background: #1a1a1a; border: 1px solid #444; padding: 2px 6px;
      border-radius: 4px; font-family: monospace; margin: 0 2px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Curaduría de frames</h1>
    <div class="stats">
      Total: <strong id="total">0</strong> |
      Conservados: <strong id="kept">0</strong> |
      <span class="discarded">Descartados: <strong id="discarded">0</strong></span>
    </div>
    <button id="exportBtn">Descargar selección</button>
    <button id="discardAllBtn" class="danger">Marcar TODOS para descartar</button>
    <button id="clearBtn" class="ghost">Limpiar marcas</button>
  </header>
  <div class="help">
    <strong>Click en miniatura</strong> = descartar / quitar marca.
    Teclas: <kbd>D</kbd> descartar actual, <kbd>K</kbd> conservar (quitar marca),
    <kbd>←</kbd> / <kbd>→</kbd> navegar, <kbd>Home</kbd> primero, <kbd>End</kbd> último.
    Tu selección se guarda automáticamente en este navegador.
  </div>
  <div id="grid" class="grid"></div>

<script>
const FRAMES = __FRAMES_JSON__;
const STORAGE_KEY = "curate_frames_v1";

let discarded = new Set(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"));
let activeIdx = 0;

const grid = document.getElementById("grid");
const totalEl = document.getElementById("total");
const keptEl = document.getElementById("kept");
const discardedEl = document.getElementById("discarded");

function updateStats() {
  totalEl.textContent = FRAMES.length;
  discardedEl.textContent = discarded.size;
  keptEl.textContent = FRAMES.length - discarded.size;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(discarded)));
}

function renderThumbs() {
  FRAMES.forEach((name, idx) => {
    const div = document.createElement("div");
    div.className = "thumb" + (discarded.has(name) ? " discarded" : "");
    div.dataset.idx = idx;
    div.dataset.name = name;
    div.innerHTML = `
      <img src="thumbs/${name}" loading="lazy" alt="${name}">
      <div class="name">${name}</div>
    `;
    div.addEventListener("click", () => toggleDiscard(idx));
    grid.appendChild(div);
  });
  setActive(0);
  updateStats();
}

function toggleDiscard(idx) {
  const name = FRAMES[idx];
  const el = grid.children[idx];
  if (discarded.has(name)) {
    discarded.delete(name);
    el.classList.remove("discarded");
  } else {
    discarded.add(name);
    el.classList.add("discarded");
  }
  updateStats();
}

function setActive(idx) {
  if (grid.children[activeIdx]) grid.children[activeIdx].classList.remove("active");
  activeIdx = Math.max(0, Math.min(FRAMES.length - 1, idx));
  const el = grid.children[activeIdx];
  if (el) {
    el.classList.add("active");
    el.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") setActive(activeIdx + 1);
  else if (e.key === "ArrowLeft") setActive(activeIdx - 1);
  else if (e.key === "Home") setActive(0);
  else if (e.key === "End") setActive(FRAMES.length - 1);
  else if (e.key.toLowerCase() === "d") {
    if (!discarded.has(FRAMES[activeIdx])) toggleDiscard(activeIdx);
  } else if (e.key.toLowerCase() === "k") {
    if (discarded.has(FRAMES[activeIdx])) toggleDiscard(activeIdx);
  }
});

document.getElementById("exportBtn").addEventListener("click", () => {
  const data = {
    total: FRAMES.length,
    kept: FRAMES.filter(n => !discarded.has(n)),
    discarded: Array.from(discarded),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "selection.json"; a.click();
  URL.revokeObjectURL(url);
});

document.getElementById("discardAllBtn").addEventListener("click", () => {
  if (!confirm("¿Marcar TODOS los frames para descartar?")) return;
  FRAMES.forEach((n, i) => {
    discarded.add(n);
    grid.children[i].classList.add("discarded");
  });
  updateStats();
});

document.getElementById("clearBtn").addEventListener("click", () => {
  discarded.clear();
  Array.from(grid.children).forEach(el => el.classList.remove("discarded"));
  updateStats();
});

renderThumbs();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    out_dir = Path(args.output_dir)
    thumbs_dir = out_dir / "thumbs"

    if not src.exists():
        raise SystemExit(f"[ERROR] No existe: {src}")

    frames = sorted([p for p in src.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not frames:
        raise SystemExit(f"[ERROR] No hay frames en {src}")

    thumbs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generando thumbnails para {len(frames)} frames...")
    for f in tqdm(frames, desc="Thumbs", unit="img"):
        out_path = thumbs_dir / f.name
        if out_path.exists():
            continue  # idempotente: no regenera si ya existe
        try:
            with Image.open(f) as im:
                im = ImageOps.exif_transpose(im)
                im.thumbnail((args.thumb_size, args.thumb_size), Image.Resampling.LANCZOS)
                im.save(out_path, "JPEG", quality=80)
        except Exception as e:
            print(f"  [AVISO] {f.name}: {e}")

    html = HTML_TEMPLATE.replace(
        "COLUMNS", str(args.columns)
    ).replace(
        "__FRAMES_JSON__", json.dumps([f.name for f in frames])
    )

    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")

    print(f"\nVisor generado en: {index_path.resolve()}")
    print(f"Thumbnails en:     {thumbs_dir.resolve()}")
    print(f"\nAbre el archivo HTML en tu navegador:")
    print(f"   file:///{index_path.resolve().as_posix()}")
    print(f"\nCuando termines de marcar descartes:")
    print(f"  1. Click en 'Descargar selección'  -> guarda selection.json")
    print(f"  2. python scripts/apply_curation.py <ruta-a-selection.json>")


if __name__ == "__main__":
    main()
