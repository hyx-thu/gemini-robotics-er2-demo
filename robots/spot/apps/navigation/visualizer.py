from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from bosdyn.api import point_cloud_pb2
from bosdyn.client import frame_helpers

from apps.navigation.registry import KeypointRegistry


DEFAULT_MAP_PATH = Path(__file__).with_name("waypoints_map.html")


MAX_POINTS_PER_SNAPSHOT = 550


def _waypoint_position(waypoint: Any) -> tuple[float, float]:
    position = waypoint.waypoint_tform_ko.position
    return position.x, position.y


def _anchor_positions(graph: Any) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for anchor in graph.anchoring.anchors:
        pose = anchor.seed_tform_waypoint.position
        positions[anchor.id] = (pose.x, pose.y)
    return positions


def _decode_xyz_32f(point_cloud: Any) -> np.ndarray:
    if point_cloud.encoding != point_cloud_pb2.PointCloud.ENCODING_XYZ_32F:
        return np.empty((0, 3), dtype=np.float32)
    if not point_cloud.data:
        return np.empty((0, 3), dtype=np.float32)
    return np.frombuffer(point_cloud.data, dtype="<f4").reshape((-1, 3))


def _snapshot_points(snapshot: Any, max_points: int = MAX_POINTS_PER_SNAPSHOT) -> list[list[float]]:
    point_cloud = snapshot.point_cloud
    points = _decode_xyz_32f(point_cloud)
    if not len(points):
        return []

    odom_tform_sensor = frame_helpers.get_a_tform_b(
        point_cloud.source.transforms_snapshot,
        frame_helpers.ODOM_FRAME_NAME,
        point_cloud.source.frame_name_sensor,
        validate=False,
    )
    if odom_tform_sensor is None:
        return []

    if len(points) > max_points:
        stride = math.ceil(len(points) / max_points)
        points = points[::stride]

    transformed = odom_tform_sensor.transform_cloud(points)
    return [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in transformed]


def graph_to_payload(
    graph: Any,
    registry: KeypointRegistry,
    snapshots: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry_names = {keypoint.waypoint_id: keypoint.name for keypoint in registry.all()}

    waypoints = []
    for waypoint in graph.waypoints:
        x, y = _waypoint_position(waypoint)
        graph_name = waypoint.annotations.name.strip()
        display_name = registry_names.get(waypoint.id) or graph_name or waypoint.id[:10]
        waypoints.append(
            {
                "id": waypoint.id,
                "name": display_name,
                "graph_name": graph_name,
                "x": x,
                "y": y,
            }
        )

    point_clouds = []
    if snapshots:
        for waypoint in graph.waypoints:
            if not waypoint.snapshot_id:
                continue
            snapshot = snapshots.get(waypoint.snapshot_id)
            if snapshot is None:
                continue
            points = _snapshot_points(snapshot)
            if points:
                point_clouds.append({"waypoint_id": waypoint.id, "points": points})

    waypoint_ids = {waypoint["id"] for waypoint in waypoints}
    edges = [
        {"from": edge.id.from_waypoint, "to": edge.id.to_waypoint}
        for edge in graph.edges
        if edge.id.from_waypoint in waypoint_ids and edge.id.to_waypoint in waypoint_ids
    ]

    return {
        "waypoints": waypoints,
        "edges": edges,
        "point_clouds": point_clouds,
    }


def write_graph_html(
    graph: Any,
    registry: KeypointRegistry,
    output: Path = DEFAULT_MAP_PATH,
    snapshots: dict[str, Any] | None = None,
) -> Path:
    payload = graph_to_payload(graph, registry, snapshots=snapshots)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_html(payload), encoding="utf-8")
    return output


def _render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    count = len(payload["waypoints"])
    edge_count = len(payload["edges"])
    point_count = sum(len(cloud["points"]) for cloud in payload["point_clouds"])
    escaped_data = html.escape(data, quote=False)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spot GraphNav Waypoints</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #666d72;
      --cloud: #2f3437;
      --line: #98a2a8;
      --node: #1f7a6d;
      --node-active: #c8442f;
      --panel: #ffffff;
      --border: #d7dbd8;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      display: grid;
      grid-template-rows: auto 1fr;
    }}

    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
    }}

    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }}

    .meta {{
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}

    main {{
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
    }}

    .map-wrap {{
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      position: relative;
    }}

    svg {{
      width: 100%;
      height: 100%;
      display: block;
      background: #fbfbf8;
    }}

    .edge {{
      stroke: var(--line);
      stroke-width: 0.25;
      stroke-linecap: round;
      opacity: 0.6;
    }}

    .cloud {{
      stroke: var(--cloud);
      stroke-width: 0.045;
      stroke-linecap: round;
      opacity: 0.32;
    }}

    .node {{
      fill: var(--node);
      stroke: white;
      stroke-width: 0.22;
      cursor: pointer;
    }}

    .node.active {{
      fill: var(--node-active);
    }}

    .label {{
      font-size: 1.65px;
      paint-order: stroke;
      stroke: #fbfbf8;
      stroke-width: 0.45px;
      stroke-linejoin: round;
      fill: #232729;
      pointer-events: none;
    }}

    aside {{
      min-height: 0;
      overflow: auto;
      background: var(--panel);
      border-left: 1px solid var(--border);
      padding: 12px;
    }}

    input {{
      width: 100%;
      height: 30px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0 10px;
      font-size: 12px;
    }}

    ul {{
      list-style: none;
      margin: 12px 0 0;
      padding: 0;
    }}

    li {{
      border-bottom: 1px solid #ecefed;
      padding: 7px 4px;
      cursor: pointer;
    }}

    li:hover, li.active {{
      background: #eef6f4;
    }}

    .name {{
      font-weight: 700;
      font-size: 12px;
    }}

    .id {{
      margin-top: 3px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 9px;
      overflow-wrap: anywhere;
    }}

    @media (max-width: 760px) {{
      main {{
        grid-template-columns: 1fr;
        grid-template-rows: 65vh 35vh;
      }}

      aside {{
        border-left: 0;
        border-top: 1px solid var(--border);
      }}
    }}
  </style>
</head>
<body>
  <script id="graph-data" type="application/json">{escaped_data}</script>
  <header>
    <h1>Spot GraphNav Waypoints</h1>
    <div class="meta">
      <span>{count} waypoints</span>
      <span>{edge_count} edges</span>
      <span>{point_count} points</span>
    </div>
  </header>
  <main>
    <section class="map-wrap">
      <svg id="map" role="img" aria-label="GraphNav waypoint map"></svg>
    </section>
    <aside>
      <input id="search" type="search" placeholder="Filter waypoints">
      <ul id="list"></ul>
    </aside>
  </main>
  <script>
    const data = JSON.parse(document.getElementById("graph-data").textContent);
    const svg = document.getElementById("map");
    const list = document.getElementById("list");
    const search = document.getElementById("search");
    const ns = "http://www.w3.org/2000/svg";
    const nodes = new Map(data.waypoints.map((w) => [w.id, w]));
    let activeId = null;

    const cloudPoints = data.point_clouds.flatMap((cloud) => cloud.points);
    const xs = data.waypoints.map((w) => w.x).concat(cloudPoints.map((p) => p[0]));
    const ys = data.waypoints.map((w) => w.y).concat(cloudPoints.map((p) => p[1]));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const pad = Math.max(maxX - minX, maxY - minY, 1) * 0.12;
    svg.setAttribute("viewBox", `${{minX - pad}} ${{-(maxY + pad)}} ${{maxX - minX + pad * 2}} ${{maxY - minY + pad * 2}}`);

    function addSvg(tag, attrs, parent = svg) {{
      const el = document.createElementNS(ns, tag);
      for (const [key, value] of Object.entries(attrs)) {{
        el.setAttribute(key, value);
      }}
      parent.appendChild(el);
      return el;
    }}

    const cloudLayer = addSvg("g", {{}});
    const edgeLayer = addSvg("g", {{}});
    const nodeLayer = addSvg("g", {{}});
    const labelLayer = addSvg("g", {{}});

    for (const cloud of data.point_clouds) {{
      const d = cloud.points.map((p) => `M ${{p[0]}} ${{-p[1]}} h 0.001`).join(" ");
      addSvg("path", {{
        class: "cloud",
        d,
      }}, cloudLayer);
    }}

    for (const edge of data.edges) {{
      const from = nodes.get(edge.from);
      const to = nodes.get(edge.to);
      if (!from || !to) continue;
      addSvg("line", {{
        class: "edge",
        x1: from.x,
        y1: -from.y,
        x2: to.x,
        y2: -to.y,
      }}, edgeLayer);
    }}

    for (const waypoint of data.waypoints) {{
      const node = addSvg("circle", {{
        class: "node",
        cx: waypoint.x,
        cy: -waypoint.y,
        r: 0.42,
        "data-id": waypoint.id,
      }}, nodeLayer);
      node.addEventListener("click", () => setActive(waypoint.id));

      addSvg("text", {{
        class: "label",
        x: waypoint.x + 0.58,
        y: -waypoint.y - 0.58,
      }}, labelLayer).textContent = waypoint.name;
    }}

    function renderList() {{
      const term = search.value.trim().toLowerCase();
      list.textContent = "";
      for (const waypoint of data.waypoints) {{
        if (term && !waypoint.name.toLowerCase().includes(term) && !waypoint.id.toLowerCase().includes(term)) {{
          continue;
        }}
        const item = document.createElement("li");
        item.dataset.id = waypoint.id;
        if (waypoint.id === activeId) item.classList.add("active");
        item.innerHTML = `<div class="name">${{waypoint.name}}</div><div class="id">${{waypoint.id}}</div>`;
        item.addEventListener("click", () => setActive(waypoint.id));
        list.appendChild(item);
      }}
    }}

    function setActive(id) {{
      activeId = id;
      document.querySelectorAll(".node").forEach((node) => node.classList.toggle("active", node.dataset.id === id));
      renderList();
      const item = list.querySelector(`[data-id="${{CSS.escape(id)}}"]`);
      if (item) item.scrollIntoView({{ block: "nearest" }});
    }}

    search.addEventListener("input", renderList);
    renderList();
  </script>
</body>
</html>
"""
