// Tiny static server for the explainer scenes (localhost only). Usage: node serve.mjs [port]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const roots = { "/three/": path.join(here, "node_modules/three/"), "/": path.join(here, "web/") };
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".css": "text/css" };

export function serve(port = 5178) {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent((req.url || "/").split("?")[0]);
    const prefix = Object.keys(roots).find((p) => url.startsWith(p));
    const base = roots[prefix];
    const file = path.normalize(path.join(base, url.slice(prefix.length) || "index.html"));
    if (!file.startsWith(base) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); res.end("not found"); return;
    }
    res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(port, "127.0.0.1", () => ok(server)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.argv[2] || 5178);
  serve(port).then(() => console.log(`scenes: http://127.0.0.1:${port}/`));
}
