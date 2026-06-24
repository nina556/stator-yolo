from __future__ import annotations

from .paths import ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    from scripts.stator_dataset_gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
