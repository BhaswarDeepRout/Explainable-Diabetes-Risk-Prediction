import requests
import json
import urllib.parse
import sys

def main():
    try:
        with open('paper/mainnnn.tex', 'r', encoding='utf-8') as f:
            tex_content = f.read()
    except Exception as e:
        print("Failed to read file:", e)
        return

    # latexonline.cc limits requests for very large files, but ours might be okay
    # Let's try downloading the compiled PDF
    print("Compiling via latexonline.cc...")
    try:
        response = requests.post(
            'https://latexonline.cc/compile',
            files={'file': ('mainnnn.tex', tex_content)}
        )
        if response.status_code == 200:
            with open('paper/mainnnn.pdf', 'wb') as f:
                f.write(response.content)
            print("Compile SUCCESS. PDF saved to paper/mainnnn.pdf")
        else:
            print("Compile FAILED.")
            print("Status:", response.status_code)
            print("Log/Output:", response.text[:2000])
    except Exception as e:
        print("Network request failed:", e)

if __name__ == "__main__":
    main()
