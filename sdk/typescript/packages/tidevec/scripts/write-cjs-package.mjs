// The packages are "type": "module"; mark the CommonJS build so Node
// loads dist/cjs/*.js with require semantics. Usage: node write-cjs-package.mjs [dir]
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";

const dir = resolve(process.argv[2] ?? "dist/cjs");
writeFileSync(resolve(dir, "package.json"), JSON.stringify({ type: "commonjs" }, null, 2) + "\n");
