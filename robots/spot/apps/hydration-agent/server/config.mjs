import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export function loadConfig() {
  const values = {};
  const paths = [
    path.join(appDirectory, ".config"),
    path.resolve(appDirectory, "..", "..", ".config"),
  ];

  for (const configPath of paths) {
    if (!existsSync(configPath)) continue;
    for (const rawLine of readFileSync(configPath, "utf8").split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#") || !line.includes("=")) continue;
      const separator = line.indexOf("=");
      const key = line.slice(0, separator).trim();
      const value = line.slice(separator + 1).trim().replace(/^["']|["']$/g, "");
      if (key) values[key] = value;
    }
  }

  return { ...values, ...process.env };
}

export { appDirectory };
