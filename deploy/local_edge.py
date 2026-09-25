import sys
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITE = Path(sys.argv[1]).resolve()
ORIGIN = sys.argv[2].rstrip("/")
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8090


def page(path: str) -> str:
    if path.endswith("/"):
        return path + "index.html"
    if "." not in path.rsplit("/", 1)[-1]:
        return path + "/index.html"
    return path


class Edge(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def _proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        request = urllib.request.Request(ORIGIN + self.path, data=body, method=self.command)
        if self.headers.get("Content-Type"):
            request.add_header("Content-Type", self.headers["Content-Type"])
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status, payload, kind = response.status, response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as error:
            status, payload, kind = error.code, error.read(), error.headers.get("Content-Type", "")
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._proxy()
        path, _, query = self.path.partition("?")
        self.path = page(path) + ("?" + query if query else "")
        return super().do_GET()

    def do_POST(self):
        return self._proxy()


ThreadingHTTPServer(("127.0.0.1", PORT), Edge).serve_forever()
