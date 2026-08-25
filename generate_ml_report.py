import markdown

input_file = r"c:\Users\gaura\.gemini\antigravity\brain\106674c5-bed4-49bb-aa7a-3b92a2bda76c\artifacts\ml_project_report.md"
output_file = r"c:\Users\gaura\OneDrive\Desktop\Gravity_Change\Applied_Machine_Learning_Project_Report.html"

with open(input_file, 'r', encoding='utf-8') as f:
    md = f.read()

html = markdown.markdown(md)

full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Applied Machine Learning Report</title>
    <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 12pt;
            line-height: 1.5;
            color: #000000;
            padding: 1in 1.5in 1in 1.5in;
            max-width: 8.5in;
            margin: auto;
            background-color: #f4f4f4;
        }}
        .page {{
            background: white;
            padding: 1in;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            min-height: 11in;
            box-sizing: border-box;
        }}
        h1 {{
            font-size: 14pt;
            font-weight: bold;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            page-break-after: avoid;
        }}
        h2, h3 {{
            font-size: 12pt;
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            page-break-after: avoid;
        }}
        p {{
            text-align: justify;
            margin-bottom: 1rem;
            margin-top: 0;
        }}
        ul, ol {{
            margin-bottom: 1rem;
            padding-left: 2rem;
        }}
        li {{
            margin-bottom: 0.5rem;
            text-align: justify;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1.5rem;
            font-size: 12pt;
        }}
        th, td {{
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }}
        caption {{
            font-size: 10pt;
            margin-bottom: 8px;
            font-weight: bold;
        }}
        .page-break {{
            page-break-after: always;
            margin: 2rem 0;
            border-bottom: 1px dashed #ccc;
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
                margin-bottom: 0;
                padding: 0;
                min-height: auto;
            }}
            .page-break {{
                border-bottom: none;
                margin: 0;
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
