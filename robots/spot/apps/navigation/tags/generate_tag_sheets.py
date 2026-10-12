from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
TAG_DIR = ROOT / "tag36h11"


def tag_id(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def render_sheet(output: Path, tag_size_mm: int, columns: int) -> None:
    tags = sorted(TAG_DIR.glob("tag36_11_*.png"), key=tag_id)
    cards = []
    for path in tags:
        ident = tag_id(path)
        cards.append(
            f"""      <section class="tag-card">
        <img src="tag36h11/{path.name}" alt="AprilTag tag36h11 ID {ident}">
        <div class="label">tag36h11 id {ident}</div>
      </section>"""
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spot AprilTag Sheet - tag36h11</title>
  <style>
    @page {{
      size: Letter;
      margin: 12mm;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #111;
    }}

    .sheet {{
      display: grid;
      grid-template-columns: repeat({columns}, {tag_size_mm}mm);
      justify-content: center;
      gap: 12mm;
      break-inside: avoid;
    }}

    .tag-card {{
      width: {tag_size_mm}mm;
      break-inside: avoid;
      page-break-inside: avoid;
      text-align: center;
    }}

    img {{
      display: block;
      width: {tag_size_mm}mm;
      height: {tag_size_mm}mm;
      image-rendering: pixelated;
      background: white;
    }}

    .label {{
      margin-top: 3mm;
      font-size: 11pt;
      font-weight: 700;
    }}

    .print-note {{
      margin: 0 0 8mm;
      font-size: 10pt;
    }}

    @media print {{
      .print-note {{
        display: none;
      }}
    }}
  </style>
</head>
<body>
  <p class="print-note">Print at 100% scale / actual size. Do not use fit-to-page scaling.</p>
  <main class="sheet">
{chr(10).join(cards)}
  </main>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> int:
    render_sheet(ROOT / "tag36h11-large-letter.html", tag_size_mm=120, columns=1)
    render_sheet(ROOT / "tag36h11-compact-letter.html", tag_size_mm=82, columns=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
