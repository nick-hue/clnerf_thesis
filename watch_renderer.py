#!/usr/bin/env python3
import sys, time, os
from http import server
import socketserver
import email.utils

WATCH_DIR = None
PORT      = None

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Render Watcher</title>
  <style>
    body { margin:0; padding:0; background:#111; }
    img  { display:block; max-width:100vw; max-height:100vh; margin:auto; }
  </style>
</head>
<body>
  <img id="render" src="/last.png" alt="latest render">
  <script>
    let lastMod = null;
    async function checkUpdate() {
      try {
        const resp = await fetch('/last.png', { method:'HEAD', cache:'no-cache' });
        const lm = resp.headers.get('Last-Modified');
        if (lm && lm !== lastMod) {
          lastMod = lm;
          // full page reload on update
          window.location.reload();
        }
      } catch(e){ console.warn(e); }
    }
    setInterval(checkUpdate, 1000);
  </script>
</body>
</html>
"""

class Handler(server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            content = INDEX_HTML.encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == '/last.png':
            img = os.path.join(WATCH_DIR, 'last.png')
            if not os.path.isfile(img):
                return self.send_error(404, "last.png not found")

            st = os.stat(img)
            lm = email.utils.formatdate(st.st_mtime, usegmt=True)

            with open(img, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Last-Modified", lm)
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_error(404)

    def do_HEAD(self):
        if self.path == '/last.png':
            img = os.path.join(WATCH_DIR, 'last.png')
            if not os.path.isfile(img):
                return self.send_error(404, "last.png not found")
            st = os.stat(img)
            lm = email.utils.formatdate(st.st_mtime, usegmt=True)
            self.send_response(200)
            self.send_header("Last-Modified", lm)
            self.end_headers()
        else:
            self.send_error(404)

def main():
    global WATCH_DIR, PORT
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} </path/to/watch_dir> <port>")
        sys.exit(1)
    WATCH_DIR, PORT = sys.argv[1], int(sys.argv[2])
    if not os.path.isdir(WATCH_DIR):
        print(f"Error: {WATCH_DIR} is not a directory")
        sys.exit(1)

    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Serving last.png from {WATCH_DIR} at http://0.0.0.0:{PORT}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")

if __name__ == '__main__':
    main()
