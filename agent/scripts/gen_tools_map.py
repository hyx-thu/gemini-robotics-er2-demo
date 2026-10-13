"""Regenerates the inline SVG flowchart of the tools map in agent/ui/index.html.

Run from agent/ (needs the tool modules):  uv run python3 gen_skills_svg.py ui/index.html
Tool hover tips are generated from the real function declarations, so they
always match what the model sees.
"""
import html
import json
import re
import sys

sys.path.insert(0, ".")
from tool import grounding_tools as _g  # pylint: disable=g-import-not-at-top
from tool import tools as _t  # pylint: disable=g-import-not-at-top

INDEX = sys.argv[1]


def _first_sentences(text, n=2, limit=260):
  text = re.sub(r"\s+", " ", text or "").strip()
  # Split on sentence ends, but not after abbreviations like "e.g." / "i.e.".
  parts = re.split(r"(?<!e\.g\.)(?<!i\.e\.)(?<=[.!?])\s+", text)
  out = " ".join(parts[:n])
  return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _decls():
  out = {}
  groups = _t.spot_sim_tools() + [{"functionDeclarations": _g.grounding_tool_declarations()}]
  for grp in groups:
    for d in grp.get("functionDeclarations", [grp]):
      params = (d.get("parameters") or {})
      req = set(params.get("required") or [])
      plist = []
      for name, spec in (params.get("properties") or {}).items():
        plist.append({
            "name": name,
            "type": str(spec.get("type", "")).lower(),
            "required": name in req,
            "desc": _first_sentences(spec.get("description", ""), 1, 120),
            "enum": list(spec.get("enum") or []),
        })
      out[d["name"]] = {"desc": _first_sentences(d.get("description", "")), "params": plist}
  return out


TIPS = _decls()

INPUTS = [('🎤 Voice', 'Microphone audio stream'), ('📹 Camera video', 'Live camera frames'),
          ('⌨️ Text', 'Typed instructions'), ('❤️ Heartbeat', 'Periodic status prompt'),
          ('📍 Location', 'Current position')]
CATS = [
    ('🐾 Locomotion', 'move around', 'drive · get_waypoints · navigate · stop', 'c-move'),
    ('🧍 Posture', 'body pose', 'stand · sit', 'c-pose'),
    ('👁️ Perception', 'see & locate', 'look · detect', 'c-see'),
    ('🦾 Manipulation', 'arm & gripper', 'pick · place · stow · wait_for_pick_up', 'c-arm'),
    ('🤝 Gestures', 'greet & interact', 'make_gesture(high_wave|shake_hand)', 'c-gest'),
    ('💬 Communication', 'talk to people', 'send_message · ack', 'c-talk'),
    ('🌐 World knowledge', 'grounded lookup', 'get_current_location · google_search · google_maps_search', 'c-web'),
]

W = 1220
o = []
A = o.append


def box(x, y, w, h, cls='fc-box'):
  A(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>')


def text(x, y, t, cls='fc-t', anchor='start'):
  A(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{html.escape(t)}</text>')


def line(pts, cls='fc-line', arrow=True):
  d = ' '.join(f'{x},{y}' for x, y in pts)
  A(f'<polyline class="{cls}" points="{d}" fill="none"'
    + (' marker-end="url(#fc-arrow)"' if arrow else '') + '/>')


# Tools column defines the vertical extent.
sx, sw, sh, sy0, sgap = 660, 470, 56, 80, 70
cy = [sy0 + i * sgap + sh / 2 for i in range(len(CATS))]
mid = (cy[0] + cy[-1]) / 2

# Inputs, centred on the same middle line.
ix, iw, ih, igap = 30, 190, 46, 66
iy0 = mid - (len(INPUTS) - 1) * igap / 2 - ih / 2
text(ix, iy0 - 14, 'INPUTS', 'fc-col')
bus_x = 262
for i, (a, b) in enumerate(INPUTS):
  y = iy0 + i * igap
  box(ix, y, iw, ih)
  text(ix + 12, y + 20, a)
  text(ix + 12, y + 37, b, 'fc-s')
  line([(ix + iw, y + ih / 2), (bus_x, y + ih / 2)], arrow=False)
line([(bus_x, iy0 + ih / 2), (bus_x, iy0 + (len(INPUTS) - 1) * igap + ih / 2)], arrow=False)

# Model.
mx, mw, mh = 318, 230, 110
my = mid - mh / 2
line([(bus_x, mid), (mx, mid)])
text((bus_x + mx) / 2, mid - 8, 'Live API', 'fc-lbl', 'middle')
box(mx, my, mw, mh, 'fc-box fc-model')
text(mx + mw / 2, my + 20, 'MODEL', 'fc-col', 'middle')
text(mx + mw / 2, my + 50, 'Gemini Robotics-ER 2', 'fc-big', 'middle')
text(mx + mw / 2, my + 72, '(alternative: Gemini 3.8 Live)', 'fc-s', 'middle')
text(mx + mw / 2, my + 94, 'reason · plan · locate in image', 'fc-s', 'middle')

# Output above the model.
ow, oh = 180, 52
ox, oy = mx + (mw - ow) / 2, my - 70 - oh
box(ox, oy, ow, oh)
text(ox + ow / 2, oy + 21, '🔊 Voice / text reply', 'fc-t', 'middle')
text(ox + ow / 2, oy + 39, 'Spoken reply & subtitles', 'fc-s', 'middle')
line([(mx + mw / 2, my), (mx + mw / 2, oy + oh)])

# Tools.
text(sx, sy0 - 14, 'ROBOT TOOLS', 'fc-col')
sbus = 616
line([(mx + mw, mid), (sbus, mid)], arrow=False)
text((mx + mw + sbus) / 2, mid - 8, 'function calls', 'fc-lbl', 'middle')
line([(sbus, cy[0]), (sbus, cy[-1])], arrow=False)
rbus = sx + sw + 30
for i, (a, b, sk, c) in enumerate(CATS):
  y = sy0 + i * sgap
  line([(sbus, cy[i]), (sx, cy[i])])
  A(f'<g class="{c}">')
  box(sx, y, sw, sh, 'fc-box fc-cat')
  A(f'<rect class="fc-bar" x="{sx}" y="{y}" width="5" height="{sh}" rx="2"/>')
  A(f'<text x="{sx + 16}" y="{y + 22}"><tspan class="fc-t">{html.escape(a)}</tspan>'
    f'<tspan class="fc-s" dx="8">{html.escape(b)}</tspan></text>')
  # One hoverable <text> per tool (monospace -> width ~= 0.6em per char).
  cw, tx = 6.95, sx + 16
  for j, token in enumerate([n.strip() for n in sk.split('·')]):
    if j:
      A(f'<text class="fc-code fc-sep" x="{tx:.1f}" y="{y + 44}">·</text>')
      tx += cw * 2
    m = re.match(r'^(\w+)(?:\((.*)\))?$', token)
    name, values = m.group(1), (m.group(2) or '')
    missing = '' if name in TIPS else ' fc-notip'
    A(f'<text class="fc-code fc-tool{missing}" data-tool="{html.escape(name)}" x="{tx:.1f}" y="{y + 44}">{html.escape(name)}</text>')
    tx += cw * (len(name) + 1)
    if values:  # e.g. the gesture names, shown as small chips after the tool
      label = '→ ' + '  ·  '.join(v.strip() for v in values.split('|'))
      A(f'<text class="fc-s fc-vals" x="{tx:.1f}" y="{y + 44}">{html.escape(label)}</text>')
      tx += 6.2 * (len(label) + 1)
  A('</g>')
  line([(sx + sw, cy[i]), (rbus, cy[i])], 'fc-loop', arrow=False)

# Closed feedback loop (polyline back into the model's bottom edge).
ybottom = sy0 + len(CATS) * sgap + 30
lx = mx + mw / 2 + 40
line([(rbus, cy[0]), (rbus, ybottom), (lx, ybottom), (lx, my + mh)], 'fc-loop')
text((lx + rbus) / 2, ybottom - 10,
     '🔁 Tool result + fresh camera frame → back to the model (closed loop)',
     'fc-loop-t', 'middle')

y0 = min(oy, iy0 - 14, sy0 - 14) - 20
h = ybottom + 24 - y0
svg = (f'<svg class="fc-svg" viewBox="0 {y0:.0f} {W} {h:.0f}" xmlns="http://www.w3.org/2000/svg" '
       'role="img" aria-label="Robot tools flow"><defs><marker id="fc-arrow" viewBox="0 0 10 10" '
       'refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" class="fc-arrowhead"/></marker></defs>' + ''.join(o) + '</svg>')

s = open(INDEX).read()
a = s.index('<svg class="fc-svg"')
b = s.index('</svg>', a) + 6
s = s[:a] + svg + s[b:]
# Tool tips JSON next to the SVG (consumed by agent-ui.js).
used = {re.match(r'^(\w+)', n.strip()).group(1) for c in CATS for n in c[2].split('·')}
tips = {k: v for k, v in TIPS.items() if k in used}
blob = '<script type="application/json" id="tool-tips">' + json.dumps(tips, ensure_ascii=False).replace('</', '<\\/') + '</script>'
s = re.sub(r'<script type="application/json" id="tool-tips">.*?</script>', '', s, flags=re.S)
i = s.index('</svg>', s.index('<svg class="fc-svg"')) + 6
s = s[:i] + blob + s[i:]
missing = sorted(used - set(TIPS))
if missing:
  print('WARNING no declaration for:', missing)
open(INDEX, 'w').write(s)
print('ok', len(svg))
