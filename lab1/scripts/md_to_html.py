#!/usr/bin/env python3
"""Minimal Markdown -> standalone HTML converter (no external deps).
Supports: headers (#.. ######), bold (**x**), inline code (`x`), fenced
code blocks (```), tables (| a | b |), horizontal rules (---), and
paragraphs. Enough for the Lab 1 report's subset of Markdown.
"""
import html
import re
import sys


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def convert(md_text):
    lines = md_text.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # fenced code block
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
            continue

        # horizontal rule
        if re.match(r"^\s*---+\s*$", line):
            out.append("<hr>")
            i += 1
            continue

        # headers
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        # table (a block of consecutive lines starting with |)
        if line.strip().startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [
                [c.strip() for c in row.strip("|").split("|")]
                for row in table_lines
            ]
            # second row is the --- separator
            header = rows[0]
            body = rows[2:] if len(rows) > 1 else []
            t = ["<table>"]
            t.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr>")
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</table>")
            out.append("\n".join(t))
            continue

        # blank line
        if line.strip() == "":
            i += 1
            continue

        # list item
        if re.match(r"^\s*-\s+", line):
            items = []
            while i < n and re.match(r"^\s*-\s+", lines[i]):
                items.append(re.match(r"^\s*-\s+(.*)$", lines[i]).group(1))
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(it)}</li>" for it in items) + "</ul>")
            continue

        numbered = re.match(r"^\s*\d+\.\s+", line)
        if numbered:
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.match(r"^\s*\d+\.\s+(.*)$", lines[i]).group(1))
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(it)}</li>" for it in items) + "</ol>")
            continue

        # paragraph (collect until blank line)
        para_lines = [line]
        i += 1
        while i < n and lines[i].strip() != "" and not lines[i].strip().startswith(("#", "|", "-", "```")):
            para_lines.append(lines[i])
            i += 1
        out.append("<p>" + inline(" ".join(para_lines)) + "</p>")

    return "\n".join(out)


TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 880px; margin: 40px auto; line-height: 1.45; color: #1a1a1a; }}
h1 {{ font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 6px; }}
h2 {{ font-size: 18px; margin-top: 28px; border-bottom: 1px solid #999; padding-bottom: 4px; }}
h3 {{ font-size: 15px; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12.5px; }}
th, td {{ border: 1px solid #999; padding: 4px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eee; }}
code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 12px; }}
pre {{ background: #f2f2f2; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 11.5px; white-space: pre-wrap; }}
pre code {{ background: none; padding: 0; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 20px 0; }}
p {{ margin: 8px 0; }}
li {{ margin: 3px 0; }}
</style>
</head><body>
{body}
</body></html>
"""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        md = f.read()
    body_html = convert(md)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.format(body=body_html))
    print(f"wrote {dst}")
