import markdown

input_file = r"c:\Users\gaura\.gemini\antigravity\brain\106674c5-bed4-49bb-aa7a-3b92a2bda76c\artifacts\ml_project_report_25_pages.md"
output_file = r"c:\Users\gaura\OneDrive\Desktop\Gravity_Change\Applied_Machine_Learning_Project_Report_Detailed.html"

with open(input_file, 'r', encoding='utf-8') as f:
    md = f.read()

html = markdown.markdown(md)

full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Detailed Applied Machine Learning Report</title>
    <style>
        /* Beautiful yet strict academic formatting */
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 12pt;
            line-height: 1.5;
            color: #000000;
            padding: 1in 1.5in 1in 1.5in;
            max-width: 8.5in;
            margin: auto;
            background-color: #f8f9fa;
        }}
        .page {{
            background: white;
            padding: 1in;
            box-shadow: 0 10px 20px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            min-height: 11in;
            box-sizing: border-box;
            border: 1px solid #e9ecef;
        }}
        
        /* Typography Rules */
        h1 {{
            font-size: 14pt;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
            page-break-after: avoid;
            color: #1a252f;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 5px;
        }}
        h2 {{
            font-size: 12pt;
            font-weight: bold;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            page-break-after: avoid;
            color: #2c3e50;
        }}
        h3 {{
            font-size: 12pt;
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            color: #34495e;
            font-style: italic;
        }}
        p {{
            text-align: justify;
            margin-bottom: 1rem;
            margin-top: 0;
            text-indent: 0.5in;
        }}
        
        /* Lists */
        ul, ol {{
            margin-bottom: 1.5rem;
            padding-left: 2.5rem;
        }}
        li {{
            margin-bottom: 0.5rem;
            text-align: justify;
        }}
        
        /* Beautiful Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            font-size: 11pt; /* Slightly smaller for tables */
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #ecf0f1;
            font-weight: bold;
            color: #2c3e50;
        }}
        caption {{
            font-size: 12pt;
            margin-bottom: 10px;
            font-weight: bold;
            text-align: center;
        }}
        
        /* Code Blocks */
        pre {{
            background: #282c34;
            color: #abb2bf;
            padding: 1.5rem;
            border-radius: 6px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 10pt;
            overflow-x: auto;
            border-left: 4px solid #3498db;
            margin: 1.5rem 0;
            white-space: pre-wrap; /* Ensure code wraps beautifully */
        }}
        code {{
            font-family: 'Courier New', Courier, monospace;
            background-color: #f1f2f6;
            padding: 2px 4px;
            border-radius: 3px;
            color: #e74c3c;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        
        /* Print Rules */
        .page-break {{
            page-break-after: always;
            margin: 3rem 0;
            border-bottom: 1px dashed #ced4da;
        }}
        
        @media print {{
            @page {{
                margin: 1in;
                size: A4;
            }}
            body {{
                background-color: white;
                padding: 0;
            }}
            .page {{
                box-shadow: none;
                border: none;
                margin-bottom: 0;
                padding: 0;
                min-height: auto;
            }}
            .page-break {{
                border-bottom: none;
                margin: 0;
            }}
            pre {{
                border: 1px solid #ccc;
                background: white;
                color: black;
            }}
            code {{
                color: black;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        {html}
    </div>
</body>
</html>
"""

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(full_html)
