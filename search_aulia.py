import urllib.request
import json
import urllib.parse

def perform_search():
    query = 'Aulia diabetes "Pima Indians" SMOTE ADASYN'
    print(f"Searching academic sources for: {query}...\n")
    
    # Using CrossRef's open API for academic papers
    url = f"https://api.crossref.org/works?query={urllib.parse.quote(query)}&select=title,author,DOI,URL,published-online,published-print,container-title&rows=10"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBot/1.0 (mailto:test@example.com)'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            items = data['message']['items']
            
            if not items:
                print("No exact matches found on Crossref.")
                return

            print("Top Results:\n" + "="*40)
            for item in items:
                title = item.get('title', ['Unknown Title'])[0]
                authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get('author', [])]
                doi = item.get('DOI', 'N/A')
                link = item.get('URL', 'N/A')
                journal = item.get('container-title', ['Unknown Journal'])[0]
                
                print(f"Title:   {title}")
                print(f"Authors: {', '.join(authors)}")
                print(f"Journal: {journal}")
                print(f"DOI:     {doi}")
                print(f"URL:     {link}")
                print("-" * 40)
                
    except Exception as e:
        print(f"Error performing search: {e}")

if __name__ == "__main__":
    perform_search()
