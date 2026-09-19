from zipfile import ZipFile
import xml.etree.ElementTree as ET
import sys

def read_docx(path):
    with ZipFile(path, 'r') as zf:
        xml_content = zf.read('word/document.xml')
    tree = ET.fromstring(xml_content)
    namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    paragraphs = []
    for p in tree.findall('.//w:p', namespace):
        texts = [node.text for node in p.findall('.//w:t', namespace) if node.text]
        if texts:
            paragraphs.append(''.join(texts))
    return '\n'.join(paragraphs)

print(read_docx(sys.argv[1] if len(sys.argv) > 1 else "paper/paper draft 1.docx"))
