import re

def md_to_html(md_text):
    html = ""
    lines = md_text.split('\n')
    
    in_table = False
    in_list = False
    
    html += "<html><head><style>table, th, td {border: 1px solid black; border-collapse: collapse; padding: 5px;}</style></head><body>\n"
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Table parsing
        if line.startswith('|'):
            if not in_table:
                html += "<table>\n"
                in_table = True
            
            # Skip separator line
            if "---" in line:
                i += 1
                continue
                
            cells = [c.strip() for c in line.split('|')[1:-1]]
            
            # First row in table should be headers if the next line is ---
            if i + 1 < len(lines) and "---" in lines[i+1]:
                html += "<tr>" + "".join(f"<th>{parse_inline(c)}</th>" for c in cells) + "</tr>\n"
            else:
                html += "<tr>" + "".join(f"<td>{parse_inline(c)}</td>" for c in cells) + "</tr>\n"
            
            i += 1
            # If next line is not a table row, end the table
            if i >= len(lines) or not lines[i].startswith('|'):
                html += "</table>\n"
                in_table = False
            continue
            
        # Headers
        header_match = re.match(r'^(#+)\s+(.*)', line)
        if header_match:
            level = len(header_match.group(1))
            html += f"<h{level}>{parse_inline(header_match.group(2))}</h{level}>\n"
            i += 1
            continue
            
        # Unordered Lists
        if re.match(r'^\*\s+', line):
            if not in_list:
                html += "<ul>\n"
                in_list = 'ul'
            text = line.strip()[2:]
            html += f"<li>{parse_inline(text)}</li>\n"
            i += 1
            if i >= len(lines) or not re.match(r'^\*\s+', lines[i]):
                html += "</ul>\n"
                in_list = False
            continue
            
        # Ordered Lists
        list_match = re.match(r'^(\d+)\.\s+(.*)', line)
        if list_match:
            if not in_list:
                html += "<ol>\n"
                in_list = 'ol'
            html += f"<li>{parse_inline(list_match.group(2))}</li>\n"
            i += 1
            if i >= len(lines) or not re.match(r'^\d+\.\s+', lines[i]):
                html += "</ol>\n"
                in_list = False
            continue
            
        # Blank lines
        if not line.strip():
            if in_table:
                html += "</table>\n"
                in_table = False
            i += 1
            continue
            
        # Regular paragraph
        html += f"<p>{parse_inline(line)}</p>\n"
        i += 1
        
    html += "</body></html>"
    return html

def parse_inline(text):
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    # Links
    text = re.sub(r'\[(.*?)\]', r'[\1]', text) # no external HTML linking, just keep braces? Let's just keep literal text.
    return text

with open("paper/paper draft 2.md", "r", encoding="utf-8") as f:
    md_text = f.read()

html = md_to_html(md_text)

with open("paper/temp.html", "w", encoding="utf-8") as f:
    f.write(html)
