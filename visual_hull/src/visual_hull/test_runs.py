from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json


@dataclass(slots=True)
class TestRun:
    root: Path
    name: str

    def path(self, filename: str) -> Path:
        return self.root / filename

    def write_json(self, filename: str, payload: object) -> Path:
        target = self.path(filename)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def write_text(self, filename: str, content: str) -> Path:
        target = self.path(filename)
        target.write_text(content, encoding="utf-8")
        return target


def create_test_run(project_root: str | Path, test_name: str) -> TestRun:
    root = Path(project_root) / "test"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = test_name.replace(" ", "-").replace("_", "-").lower()
    run_root = root / f"{timestamp}-{slug}"
    suffix = 1
    while run_root.exists():
        suffix += 1
        run_root = root / f"{timestamp}-{slug}-{suffix}"
    run_root.mkdir(parents=True, exist_ok=False)
    return TestRun(root=run_root, name=test_name)


def write_report_markdown(run: TestRun, title: str, summary_lines: list[str]) -> Path:
    body = [f"# {title}", ""]
    body.extend(f"- {line}" for line in summary_lines)
    body.append("")
    return run.write_text("report.md", "\n".join(body))
