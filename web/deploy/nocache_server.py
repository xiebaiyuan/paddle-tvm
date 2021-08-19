import http.server
import sys

port = int(sys.argv[1])

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'max-age=0')
        self.send_header('Expires', '0')
        super().end_headers()

httpServer = http.server.HTTPServer(('', port), NoCacheHTTPRequestHandler)
httpServer.serve_forever()