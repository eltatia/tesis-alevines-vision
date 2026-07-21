"""
fix_labelimg.py
---------------
Arregla 3 problemas de labelImg en Windows con Python 3.12+:

  1. CRASH AL ABRIR: labelImg.py hace `import distutils.spawn`, y distutils
     fue ELIMINADO de Python 3.12. Como en realidad nunca usa el módulo,
     se elimina el import. (Antes: la ventana se cerraba al instante.)

  2. ZOOM CON CTRL+RUEDA: usaba `Qt.ControlModifier == int(mods)`, una
     comparación frágil que falla si hay otros modificadores o en PyQt5
     reciente. Se cambia por máscara de bits `(mods & Qt.ControlModifier)`.

  3. PANEO CON EL MOUSE: solo se podía desplazar con clic izquierdo sobre
     zona vacía (casi imposible con decenas de cajas). Se añade paneo con
     el BOTÓN CENTRAL (rueda presionada), que es lo estándar y no choca
     con dibujar/seleccionar.

Actualiza tanto los archivos _parches/ de cada lote como los paquetes ya
instalados en los venv locales, para que el arreglo aplique de inmediato.

Uso:
    python scripts/fix_labelimg.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------- parche 1: quitar distutils ----------
def fix_labelimg_py(text: str) -> tuple[str, bool]:
    old = "import distutils.spawn\n"
    if old in text:
        return text.replace(old, ""), True
    return text, False


# ---------- parche 2 y 3: canvas ----------
ZOOM_OLD = "        if Qt.ControlModifier == int(mods) and v_delta:"
ZOOM_NEW = "        if (mods & Qt.ControlModifier) and v_delta:"

PAN_PRESS_OLD = """    def mousePressEvent(self, ev):
        pos = self.transform_pos(ev.pos())

        if ev.button() == Qt.LeftButton:"""
PAN_PRESS_NEW = """    def mousePressEvent(self, ev):
        pos = self.transform_pos(ev.pos())

        # Paneo con BOTON CENTRAL (rueda presionada): mover la imagen
        if ev.button() == Qt.MiddleButton:
            self.pan_initial_pos = pos
            QApplication.setOverrideCursor(QCursor(Qt.ClosedHandCursor))
            return

        if ev.button() == Qt.LeftButton:"""

PAN_MOVE_OLD = """        # Update coordinates in status bar if image is opened
        window = self.parent().window()
        if window.file_path is not None:
            self.parent().window().label_coordinates.setText(
                'X: %d; Y: %d' % (pos.x(), pos.y()))
"""
PAN_MOVE_NEW = """        # Update coordinates in status bar if image is opened
        window = self.parent().window()
        if window.file_path is not None:
            self.parent().window().label_coordinates.setText(
                'X: %d; Y: %d' % (pos.x(), pos.y()))

        # Paneo con BOTON CENTRAL mantenido
        if ev.buttons() & Qt.MiddleButton:
            delta_x = pos.x() - self.pan_initial_pos.x()
            delta_y = pos.y() - self.pan_initial_pos.y()
            self.scrollRequest.emit(delta_x, Qt.Horizontal)
            self.scrollRequest.emit(delta_y, Qt.Vertical)
            self.update()
            return
"""

PAN_RELEASE_OLD = """    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:"""
PAN_RELEASE_NEW = """    def mouseReleaseEvent(self, ev):
        # Fin del paneo con boton central
        if ev.button() == Qt.MiddleButton:
            QApplication.restoreOverrideCursor()
            return

        if ev.button() == Qt.RightButton:"""


def fix_canvas_py(text: str) -> tuple[str, list[str]]:
    done = []
    if ZOOM_OLD in text:
        text = text.replace(ZOOM_OLD, ZOOM_NEW)
        done.append("zoom Ctrl+rueda")
    if PAN_PRESS_OLD in text and "BOTON CENTRAL" not in text:
        text = text.replace(PAN_PRESS_OLD, PAN_PRESS_NEW)
        text = text.replace(PAN_MOVE_OLD, PAN_MOVE_NEW, 1)
        text = text.replace(PAN_RELEASE_OLD, PAN_RELEASE_NEW, 1)
        done.append("paneo boton central")
    return text, done


def patch_file(path: Path, kind: str) -> str:
    if not path.exists():
        return f"  [skip] no existe: {path}"
    text = path.read_text(encoding="utf-8")
    if kind == "labelimg":
        new, changed = fix_labelimg_py(text)
        if changed:
            path.write_text(new, encoding="utf-8")
            return f"  [OK] {path.name}: quitado import distutils"
        return f"  [--] {path.name}: ya estaba bien"
    else:
        new, done = fix_canvas_py(text)
        if done:
            path.write_text(new, encoding="utf-8")
            return f"  [OK] {path.name}: {', '.join(done)}"
        return f"  [--] {path.name}: ya estaba bien"


def main() -> None:
    targets: list[tuple[Path, str]] = []

    # 1) archivos fuente _parches de cada lote
    for parches in ROOT.glob("data/labeling_collab/*/_parches"):
        targets.append((parches / "labelImg.py", "labelimg"))
        targets.append((parches / "canvas.py", "canvas"))

    # 2) paquetes ya instalados en los venv (para que aplique YA)
    for venv in ROOT.glob("data/labeling_collab/*/venv"):
        targets.append((venv / "Lib/site-packages/labelImg/labelImg.py", "labelimg"))
        targets.append((venv / "Lib/site-packages/libs/canvas.py", "canvas"))
    # venv principal del proyecto, si tiene labelImg
    targets.append((ROOT / "venv/Lib/site-packages/labelImg/labelImg.py", "labelimg"))
    targets.append((ROOT / "venv/Lib/site-packages/libs/canvas.py", "canvas"))

    print("Aplicando arreglos de labelImg...\n")
    for path, kind in targets:
        print(patch_file(path, kind))
    print("\nListo. Controles: Ctrl+rueda = zoom | boton central arrastrando = mover imagen")


if __name__ == "__main__":
    main()
