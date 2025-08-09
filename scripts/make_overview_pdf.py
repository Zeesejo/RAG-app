from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
import os

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DOC_MD = os.path.join(ROOT, 'docs', 'Litends-Lab-RAG-App-Overview.md')
OUT_PDF = os.path.join(ROOT, 'docs', 'Litends-Lab-RAG-App-Overview.pdf')
LOGO = os.path.join(ROOT, 'assets', 'litends-logo-jul18-2nd-edition.jpg')

# Very simple Markdown-to-Paragraph converter for headings and paragraphs

def render_md_to_flowables(md_text: str):
    styles = getSampleStyleSheet()
    story = []
    for raw in md_text.splitlines():
        line = raw.strip('\n')
        if not line:
            story.append(Spacer(1, 0.2*inch))
            continue
        if line.startswith('# '):
            story.append(Paragraph(f"<b>{line[2:].strip()}</b>", styles['Title']))
        elif line.startswith('## '):
            story.append(Paragraph(f"<b>{line[3:].strip()}</b>", styles['Heading2']))
        elif line.startswith('### '):
            story.append(Paragraph(f"<b>{line[4:].strip()}</b>", styles['Heading3']))
        else:
            # basic escaping
            safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe, styles['BodyText']))
    return story


def main():
    os.makedirs(os.path.join(ROOT, 'docs'), exist_ok=True)
    with open(DOC_MD, 'r', encoding='utf-8') as f:
        md = f.read()

    doc = SimpleDocTemplate(OUT_PDF, pagesize=LETTER, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    if os.path.exists(LOGO):
        story.append(Image(LOGO, width=2.0*inch, height=2.0*inch))
        story.append(Spacer(1, 0.2*inch))

    story.extend(render_md_to_flowables(md))

    doc.build(story)
    print(f"Wrote {OUT_PDF}")


if __name__ == '__main__':
    main()
