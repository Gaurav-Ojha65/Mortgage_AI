import re
from pathlib import Path

readme_path = Path('README.md')
content = readme_path.read_text(encoding='utf-8')

# Check links and images
links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)

print(f'Total Links: {len(links)}')
for text, url in links:
    if url.startswith('http') or url.startswith('#'):
        continue
    clean_url = url.split('#')[0]
    if not Path(clean_url).exists():
        print(f'  [BROKEN LINK] [{text}]({url}) -> {clean_url} does not exist!')
    else:
        print(f'  [VALID LINK] [{text}]({url})')

print(f'\nTotal Images: {len(images)}')
for alt, src in images:
    if src.startswith('http'):
        continue
    if not Path(src).exists():
        print(f'  [BROKEN IMAGE] ![{alt}]({src}) -> {src} does not exist!')
    else:
        print(f'  [VALID IMAGE] ![{alt}]({src})')

# Keyword audit
keywords = ['file:///', 'v3.0', 'v2.1', '0.055', '0.405', 'payment_history_score', '94.2', '99.9%', '1000+', 'production-grade', 'enterprise-grade']
print('\nKeyword Audit:')
for kw in keywords:
    matches = [m.start() for m in re.finditer(re.escape(kw), content, re.IGNORECASE)]
    print(f'  "{kw}": {len(matches)} matches')
