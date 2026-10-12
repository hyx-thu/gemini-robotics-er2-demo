"""Tests for OpenAPI-driven Spot tools."""

from __future__ import annotations

import unittest

import httpx

from embodiment.spot import openapi_tools
from embodiment.spot import robot_client


OPENAPI_DOCUMENT = {
    "paths": {
        "/navigate": {
            "post": {
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/NavigateRequest"}
                        }
                    },
                }
            }
        },
        "/battery": {"get": {}},
        "/connect": {"post": {}},
    },
    "components": {
        "schemas": {
            "NavigateRequest": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "title": "Name"},
                    "take_lease": {"type": "boolean", "default": False},
                    "max_distance": {
                        "anyOf": [{"type": "number"}, {"type": "null"}]
                    },
                },
                "required": ["name"],
            }
        }
    },
}


DETECT_DOCUMENT = {
    "paths": {
        "/detect": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DetectRequest"}
                        }
                    }
                }
            }
        },
        "/detect/pick-target": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/DetectPickTargetRequest"
                            }
                        }
                    }
                }
            }
        },
        "/arm/approach": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ApproachRequest"}
                        }
                    }
                }
            }
        },
    },
    "components": {
        "schemas": {
            "DetectRequest": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "title": "Instruction"},
                    "color_source": {
                        "type": "string",
                        "title": "Color Source",
                        "default": "hand_color_image",
                    },
                    "depth_source": {
                        "type": "string",
                        "title": "Depth Source",
                        "default": "hand_depth_in_hand_color_frame",
                    },
                    "point_cloud_stride": {
                        "type": "integer",
                        "title": "Point Cloud Stride",
                        "default": 4,
                    },
                },
                "required": ["instruction"],
            },
            "DetectPickTargetRequest": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "title": "Instruction"},
                },
                "required": ["instruction"],
            },
            "ApproachRequest": {
                "type": "object",
                "properties": {
                    "pose": {"$ref": "#/components/schemas/PoseRequest"},
                    "standoff_m": {
                        "type": "number",
                        "title": "Standoff M",
                        "default": 0.0,
                    },
                },
                "required": ["pose"],
            },
            "PoseRequest": {
                "type": "object",
                "properties": {
                    "frame_name": {
                        "type": "string",
                        "title": "Frame Name",
                        "default": "vision",
                    },
                    "x": {"type": "number", "title": "X"},
                    "y": {"type": "number", "title": "Y"},
                    "z": {"type": "number", "title": "Z"},
                },
                "required": ["x", "y", "z"],
            },
        }
    },
}


class OpenApiToolsTest(unittest.TestCase):

  def test_builds_allowlisted_resolved_declarations(self):
    declarations, operations = openapi_tools.build_openapi_tools(
        OPENAPI_DOCUMENT
    )
    by_name = {declaration["name"]: declaration for declaration in declarations}

    self.assertEqual({"navigate", "get_battery"}, set(by_name))
    self.assertNotIn("connect", operations)
    navigate = by_name["navigate"]
    self.assertEqual(["name"], navigate["parameters"]["required"])
    self.assertEqual(
        "NUMBER", navigate["parameters"]["properties"]["max_distance"]["type"]
    )
    self.assertEqual("/navigate", operations["navigate"].path)

  def test_exposes_only_lightweight_detect_and_documents_nested_pose_units(self):
    declarations, _ = openapi_tools.build_openapi_tools(DETECT_DOCUMENT)
    by_name = {declaration["name"]: declaration for declaration in declarations}

    self.assertIn("detect", by_name)
    self.assertEqual(
        ["instruction"], by_name["detect"]["parameters"]["required"]
    )
    self.assertEqual(
        {"instruction"},
        set(by_name["detect"]["parameters"]["properties"]),
    )

    pose = by_name["approach_pose"]["parameters"]["properties"]["pose"]
    self.assertIn("vision", pose["properties"]["frame_name"]["description"])
    self.assertIn("meters", pose["properties"]["x"]["description"])

  def test_every_allowlisted_tool_has_substantive_description(self):
    for policy in openapi_tools.TOOL_POLICIES.values():
      with self.subTest(tool=policy.name):
        self.assertGreaterEqual(len(policy.description), 80)


class SpotRobotClientTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    self.requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
      self.requests.append(request)
      if request.url.path == "/openapi.json":
        return httpx.Response(200, json=OPENAPI_DOCUMENT)
      if request.url.path == "/navigate":
        return httpx.Response(200, json={"status": "arrived"})
      return httpx.Response(404)

    self.client = robot_client.SpotRobotClient(base_url="http://spot.test")
    await self.client._client.aclose()  # pylint: disable=protected-access
    self.client._client = httpx.AsyncClient(  # pylint: disable=protected-access
        base_url="http://spot.test", transport=httpx.MockTransport(handler)
    )

  async def asyncTearDown(self):
    await self.client.close()

  async def test_loads_contract_and_dispatches_json_body(self):
    declarations = await self.client.load_openapi_tools()
    result = await self.client.execute_openapi_action(
        "navigate", name="home1", take_lease=True
    )

    self.assertEqual(2, len(declarations))
    self.assertEqual({"status": "arrived"}, result)
    request = self.requests[-1]
    self.assertEqual("POST", request.method)
    self.assertEqual("/navigate", request.url.path)
    self.assertEqual(
        {"name": "home1", "take_lease": True},
        __import__("json").loads(request.content),
    )


if __name__ == "__main__":
  unittest.main()
