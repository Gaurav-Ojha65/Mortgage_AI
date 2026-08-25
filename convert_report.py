import markdown

input_file = r"c:\Users\gaura\.gemini\antigravity\brain\106674c5-bed4-49bb-aa7a-3b92a2bda76c\artifacts\capstone_project_report_premium.md"
output_file = r"c:\Users\gaura\OneDrive\Desktop\Gravity_Change\capstone_project_report_premium.html"

with open(input_file, 'r', encoding='utf-8') as f:
    md = f.read()

html = markdown.markdown(md)

full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Premium Capstone Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');
        
        :root {{
            --primary: #1e3799;
            --secondary: #4a69bd;
            --accent: #079992;
            --text-dark: #2f3640;
            --text-light: #7f8fa6;
            --bg-color: #f1f2f6;
            --cover-green: #e8f8f5; /* Light green requested by user */
        }}
        
        body {{
            /* We use Times New Roman as requested, falling back to Lora for elegance if missing */
            font-family: 'Times New Roman', 'Lora', serif;
            font-size: 12pt;
            line-height: 1.7;
            color: var(--text-dark);
            background-color: var(--bg-color);
            margin: 0;
            padding: 2rem;
        }}

        /* Page Simulation */
        .page-break {{
            page-break-after: always;
            border-bottom: 2px dashed #dcdde1;
            margin: 4rem 0;
        }}
        
        .document-section {{
            background: white;
            padding: 1in 1in 1in 1.5in; /* Standard margins */
            max-width: 8.5in;
            margin: 0 auto 2rem auto;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            min-height: 11in;
            box-sizing: border-box;
        }}

        /* Cover Page Specific */
        .cover-page {{
            background: var(--cover-green);
            padding: 1in 1in 1in 1.5in;
            max-width: 8.5in;
            margin: 0 auto;
            min-height: 11in;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            text-align: center;
            box-sizing: border-box;
            border: 4px double #1abc9c;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .cover-header {{
            font-family: 'Playfair Display', serif;
            font-size: 18pt;
            letter-spacing: 2px;
            color: var(--primary);
            margin-top: 2rem;
        }}
        
        .main-title {{
            font-family: 'Playfair Display', serif;
            font-size: 28pt;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
            letter-spacing: 1px;
        }}
        
        .sub-title {{
            font-size: 14pt;
            font-style: italic;
            color: var(--secondary);
            margin-top: 0;
        }}

        /* Typography & Content */
        h1.chapter-title {{
            font-family: 'Playfair Display', serif;
            font-size: 22pt;
            color: var(--primary);
            border-bottom: 3px solid var(--accent);
            padding-bottom: 10px;
            margin-top: 0;
            text-transform: uppercase;
        }}
        
        h2.section-title {{
            font-family: 'Playfair Display', serif;
            font-size: 18pt;
            color: var(--primary);
            margin-bottom: 2rem;
            letter-spacing: 1px;
        }}
        
        h2 {{
            font-size: 14pt;
            color: var(--secondary);
            margin-top: 2rem;
            text-transform: uppercase;
        }}
        
        h3 {{
            font-size: 12pt;
            color: var(--text-dark);
            margin-top: 1.5rem;
        }}

        p.human-text {{
            text-align: justify;
            margin-bottom: 1.5rem;
            text-indent: 1.5rem; /* Traditional academic indent */
        }}

        p.drop-cap::first-letter {{
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: bold;
            float: left;
            margin-top: 0.1rem;
            margin-bottom: -0.5rem;
            margin-right: 0.5rem;
            line-height: 1;
            color: var(--accent);
        }}

        .text-center {{
            text-align: center;
        }}

        /* Elegant Lists */
        ul.elegant-list {{
            list-style: none;
            padding-left: 0;
        }}
        
        ul.elegant-list li {{
            position: relative;
            padding-left: 2rem;
            margin-bottom: 1rem;
            text-align: justify;
        }}
        
        ul.elegant-list li::before {{
            content: "✦";
            position: absolute;
            left: 0;
            color: var(--accent);
            font-size: 1.2rem;
        }}

        /* Grids & Tables */
        .student-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 2rem auto;
            width: 80%;
            text-align: center;
        }}
        
        .signature-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 3rem;
            margin-top: 4rem;
        }}
        
        .sig-box {{
            border-top: 1px solid #ccc;
            text-align: center;
            padding-top: 0.5rem;
        }}

        /* Table of Contents */
        .toc-list {{
            list-style: none;
            padding: 0;
        }}
        
        .toc-list li {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.8rem;
            border-bottom: 1px dotted #ccc;
        }}
        
        .toc-title {{
            background: white;
            padding-right: 10px;
            margin-bottom: -5px; /* Pull down over border */
        }}
        
        .toc-title.indent {{
            padding-left: 2rem;
            color: #555;
        }}
        
        .toc-page {{
            background: white;
            padding-left: 10px;
            margin-bottom: -5px;
        }}
        
        .toc-spacer {{
            height: 20px;
            border: none !important;
        }}

        /* Code Blocks */
        pre {{
            background: #282c34;
            color: #abb2bf;
            padding: 1.5rem;
            border-radius: 8px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 10pt;
            overflow-x: auto;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
            border-left: 5px solid var(--accent);
        }}
        
        code {{
            font-family: 'Courier New', Courier, monospace;
        }}

        /* Print CSS for PDF */
        @media print {{
            @page {{
                margin: 0; /* Let the body margins handle it to preserve background colors */
                size: A4;
            }}
            body {{
                background: white;
                padding: 0;
            }}
            .cover-page {{
                margin: 0;
                box-shadow: none;
                border: none;
                padding: 1in 1in 1in 1.5in;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .document-section {{
                margin: 0;
                box-shadow: none;
                padding: 1in 1in 1in 1.5in;
                page-break-after: always;
            }}
            .page-break {{
                display: none; /* handled by section page-breaks */
            }}
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(full_html)
