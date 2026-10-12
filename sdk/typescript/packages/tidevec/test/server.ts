// Spawns a throwaway tidevec-server for integration tests.
//
// The binary comes from $TIDEVEC_SERVER_BIN, else <repo>/build-eco or
// <repo>/build, else $PATH. When none exists the suites are skipped;
// set TIDEVEC_REQUIRE_SERVER=1 (as CI should) to fail instead.
import { type ChildProcess, spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(fileURLToPath(new URL(".", import.meta.url)), "../../../../..");

export function serverBinary(): string | undefined {
  const candidates = [
    process.env.TIDEVEC_SERVER_BIN,
    join(REPO, "build-eco", "tidevec-server"),
    join(REPO, "build", "tidevec-server"),
    ...(process.env.PATH ?? "").split(delimiter).map((d) => join(d, "tidevec-server")),
  ];
  const found = candidates.find((c) => c && existsSync(c));
  if (!found && process.env.TIDEVEC_REQUIRE_SERVER === "1") {
    throw new Error("tidevec-server not found and TIDEVEC_REQUIRE_SERVER=1");
  }
  return found;
}

function freePort(): Promise<number> {
  return new Promise((ok, fail) => {
    const s = createServer();
    s.once("error", fail);
    s.listen(0, "127.0.0.1", () => {
      const addr = s.address();
      s.close(() => (addr && typeof addr === "object" ? ok(addr.port) : fail(new Error("no port"))));
    });
  });
}

export interface RunningServer {
  url: string;
  stop(): Promise<void>;
}

export async function startServer(extraArgs: string[] = []): Promise<RunningServer> {
  const bin = serverBinary();
  if (!bin) throw new Error("tidevec-server not found");
  const port = await freePort();
  const dataDir = mkdtempSync(join(tmpdir(), "tidevec-ts-"));
  const args = ["--port", String(port), "--data-dir", dataDir, "--quiet", ...extraArgs];
  if (!extraArgs.includes("--api-key")) args.push("--no-require-auth");
  const proc: ChildProcess = spawn(bin, args, { stdio: "ignore" });
  const url = `http://127.0.0.1:${port}`;

  let exited = false;
  proc.once("exit", () => (exited = true));
  for (let i = 0; i < 100; i++) {
    if (exited) throw new Error(`tidevec-server exited with code ${proc.exitCode}`);
    try {
      const r = await fetch(`${url}/health`);
      if (r.ok) break;
    } catch {
      // not listening yet
    }
    await new Promise((r) => setTimeout(r, 100));
    if (i === 99) {
      proc.kill("SIGKILL");
      throw new Error("tidevec-server did not start");
    }
  }

  return {
    url,
    async stop() {
      if (!exited) {
        const done = new Promise((r) => proc.once("exit", r));
        proc.kill("SIGTERM");
        const timer = setTimeout(() => proc.kill("SIGKILL"), 10_000);
        await done;
        clearTimeout(timer);
      }
      rmSync(dataDir, { recursive: true, force: true });
    },
  };
}

export const hasServer = Boolean(serverBinary());
