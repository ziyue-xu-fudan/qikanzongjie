import pandas as pd
import re
import os

def simple_parse_bmj():
    input_file = "/Users/ziyuexu/Documents/trae_projects/paper1/abstract-BMJJournal-set (2).txt"
    output_file = "/Users/ziyuexu/Documents/trae_projects/paper1/BMJ.xlsx"
    
    print("🔄 Simple parsing BMJ txt...")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by article number (e.g., "1. BMJ.", "2. BMJ.")
    articles = re.split(r'\n\n\d+\.\s*BMJ\.', content)
    
    data = []
    for raw_text in articles:
        if not raw_text.strip(): continue
        
        row = {}
        
        # 1. PMID
        pmid_match = re.search(r'PMID:\s*(\d+)', raw_text)
        if pmid_match:
            row['PMID'] = pmid_match.group(1)
        
        # 2. Title (usually at the beginning or after doi)
        # Try to find the title. It's often the first substantial text block.
        # Simplistic approach: take the first 200 chars as title candidate if no better way
        # But looking at previous parser: r'\.\s*doi:\s*[\d\.-]+/[\w\.-]+\.\s*\n\n(.+?)\n\n'
        doi_match = re.search(r'doi:\s*[\d\.-]+/[\w\.-]+\.\s*\n\n(.+?)\n\n', raw_text, re.DOTALL)
        if doi_match:
            row['Title'] = doi_match.group(1).strip().replace('\n', ' ')
        else:
            # Fallback: take first non-empty line
            lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
            if lines:
                row['Title'] = lines[0] # Very rough approximation
        
        # 3. Abstract
        # Combine sections like OBJECTIVE, DESIGN, etc.
        abstract_parts = []
        keywords = ['OBJECTIVE', 'DESIGN', 'SETTING', 'PARTICIPANTS', 'MAIN OUTCOME MEASURES', 'RESULTS', 'CONCLUSIONS']
        
        for kw in keywords:
            # Find keyword and text until next keyword or end
            pattern = f"{kw}:\\s*(.+?)(?={'|'.join(keywords)}|©|Conflict|$)"
            # This regex is tricky to construct dynamically correctly, let's do simple search
            match = re.search(rf'{kw}:\s*(.+?)(?=\n[A-Z]+:|©|$)', raw_text, re.DOTALL)
            if match:
                abstract_parts.append(f"{kw}: {match.group(1).strip().replace(chr(10), ' ')}")
        
        if abstract_parts:
            row['Abstract'] = " ".join(abstract_parts)
        else:
            # If no structured abstract, try to grab the big chunk of text?
            # Maybe just leave it empty and let AI fetch it?
            # Or try to find 'Abstract' keyword?
            pass
            
        # Add to list if we at least have PMID or Title
        if row.get('PMID') or row.get('Title'):
            data.append(row)
            
    df = pd.DataFrame(data)
    print(f"📊 Extracted {len(df)} articles.")
    
    if not df.empty:
        # Save
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Filtered_Literature', index=False)
        print(f"✅ Saved clean BMJ.xlsx with {len(df)} rows.")
    else:
        print("❌ Failed to extract any data.")

if __name__ == "__main__":
    simple_parse_bmj()
