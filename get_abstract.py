import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

try:
    req = urllib.request.Request("https://api.crossref.org/works/10.1109/icicos68590.2025.11329981", headers={'User-Agent': 'Testing/1.0'})
    with urllib.request.urlopen(req, context=ctx) as r:
        data = json.loads(r.read().decode())['message']
        print(f"Abstract: {data.get('abstract', 'No abstract')}")
except Exception as e:
    print(e)
