import http.server
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
port = int(os.environ.get('PORT', 5000))
http.server.HTTPServer(('0.0.0.0', port), http.server.SimpleHTTPRequestHandler).serve_forever()
