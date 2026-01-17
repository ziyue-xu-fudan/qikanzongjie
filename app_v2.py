import streamlit as st
import pandas as pd
import os
import re

# --- Config ---
st.set_page_config(page_title="Screening V2", layout="wide", page_icon="🔬")
FILE_PATH = "Literature_Screening_List.xlsx"

# --- Helper Functions ---
def load_data():
    if not os.path.exists(FILE_PATH):
        st.error(f"File not found: {FILE_PATH}")
        return pd.DataFrame()
    # Read ensuring string types for critical columns
    df = pd.read_excel(FILE_PATH, dtype=str)
    df = df.fillna("")
    return df

def save_data(df):
    df.to_excel(FILE_PATH, index=False)

def format_abstract_structure(text):
    """
    Adds line breaks and bold formatting to abstract sections.
    Should be called AFTER text highlighting to avoid breaking HTML tags.
    """
    if not text:
        return text
        
    # Common section headers
    sections = [
        "BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS", "FINDINGS", "INTERPRETATION", 
        "OBJECTIVE", "DESIGN", "SETTING", "PARTICIPANTS", "MAIN OUTCOMES AND MEASURES",
        "IMPORTANCE", "FUNDING"
    ]
    
    # Replace "SECTION: " with "<br><b>SECTION: </b>"
    # Using regex to match case-insensitive but usually these are UPPERCASE in PubMed
    for section in sections:
        # Improved Regex:
        # 1. (^|[\.\!\?]\s+) -> Start of string OR End of previous sentence (dot/bang/qmark + space)
        # 2. (\bSECTION\b) -> The section name (word boundary)
        # 3. \s*: -> Optional space followed by Colon (Mandatory to identify as header)
        pattern = re.compile(r"(^|[\.\!\?]\s+)(\b" + re.escape(section) + r"\b\s*:)", re.IGNORECASE)
        
        def replace_func(match):
            prefix = match.group(1) if match.group(1) else ""
            # The captured section header (including colon)
            header_full = match.group(2)
            
            # Normalize to UPPERCASE for the keyword part, keep colon
            # Find where the colon is
            colon_idx = header_full.find(':')
            if colon_idx != -1:
                kw = header_full[:colon_idx].upper()
                rest = header_full[colon_idx:]
                header_html = f"<span style='font-weight:900; color:#2c3e50; font-size:1.05em'>{kw}</span>{rest}"
            else:
                header_html = header_full.upper() # Fallback
                
            return f"{prefix}<br><br>{header_html} "
            
        text = pattern.sub(replace_func, text)
        
    return text

def highlight_text(text, search_keyword):
    if not text:
        return text
        
    # 1. Highlight Default Cancer Keywords (Red)
    default_keywords = ["Cancer", "Tumor", "Tumour", "Oncology", "Neoplasm", "Carcinoma", "Malignancy"]
    for kw in default_keywords:
        pattern = re.compile(f"({re.escape(kw)})", re.IGNORECASE)
        text = pattern.sub(r'<span style="background-color: #ffcccc; color: #cc0000; font-weight: bold; padding: 0 2px;">\1</span>', text)
        
    # 2. Highlight User Search Keyword (Orange/Yellow)
    if search_keyword:
        pattern = re.compile(f"({re.escape(search_keyword)})", re.IGNORECASE)
        text = pattern.sub(r'<span style="background-color: #fff3cd; color: #856404; font-weight: bold; border-bottom: 2px solid #ffc107;">\1</span>', text)
        
    return text

# --- Main App ---
def main():
    # 1. Load Data
    if 'df' not in st.session_state:
        st.session_state.df = load_data()
    
    df = st.session_state.df
    total_count = len(df)
    
    # --- Top Metrics Bar ---
    included = len(df[df["Select? (Y/N)"] == "Y"])
    excluded = len(df[df["Select? (Y/N)"] == "N"])
    pending = total_count - included - excluded
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Total Papers", total_count)
    c2.metric("⏳ Pending", pending)
    c3.metric("✅ Included", included)
    c4.metric("❌ Excluded", excluded)
    
    st.divider()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.title("Filters")
        
        # Status Filter
        status_filter = st.radio("Show Status:", ["All", "Pending Only", "Included Only", "Excluded Only"])
        
        # Cancer Type Filter
        all_cancers = sorted([str(x) for x in df['Cancer_Type'].unique() if x])
        all_cancers = [c for c in all_cancers if c and c != "nan"]
        selected_cancers = st.multiselect("Filter by Cancer Type:", all_cancers)
        
        # Phase / Design Filter
        all_phases = sorted([str(x) for x in df['Phase'].unique() if x])
        all_phases = [p for p in all_phases if p and p != "nan"]
        selected_phases = st.multiselect("Filter by Phase:", all_phases)
        
        # Study Design Filter
        if 'Study_Design' in df.columns:
            all_designs = sorted([str(x) for x in df['Study_Design'].unique() if x])
            all_designs = [d for d in all_designs if d and d != "nan"]
            selected_designs = st.multiselect("Filter by Study Design:", all_designs)
        else:
            selected_designs = []
        
        # Keyword Search
        search_query = st.text_input("Search (Title/Abstract/AI):", "")
        
        st.divider()
        
        # Actions
        if st.button("💾 Force Save to Disk"):
            save_data(df)
            st.success("Saved!")
            
        if st.button("🔄 Reload from Disk"):
            st.session_state.df = load_data()
            st.rerun()
            
        st.divider()
        
        # Download
        if included > 0:
            st.write("### Export Included")
            csv = df[df["Select? (Y/N)"] == "Y"].to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "included_papers.csv", "text/csv")

    # --- Filtering Logic ---
    filtered_df = df.copy()
    
    # 1. Status Filter
    if status_filter == "Pending Only":
        filtered_df = filtered_df[~filtered_df["Select? (Y/N)"].isin(["Y", "N"])]
    elif status_filter == "Included Only":
        filtered_df = filtered_df[filtered_df["Select? (Y/N)"] == "Y"]
    elif status_filter == "Excluded Only":
        filtered_df = filtered_df[filtered_df["Select? (Y/N)"] == "N"]
        
    # 2. Cancer Filter
    if selected_cancers:
        filtered_df = filtered_df[filtered_df["Cancer_Type"].isin(selected_cancers)]
        
    # 3. Phase Filter
    if selected_phases:
        filtered_df = filtered_df[filtered_df["Phase"].isin(selected_phases)]
        
    # 4. Study Design Filter
    if selected_designs and 'Study_Design' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Study_Design"].isin(selected_designs)]
        
    # 5. Search Filter
    if search_query:
        q = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["Title"].str.lower().str.contains(q) | 
            filtered_df["Abstract"].str.lower().str.contains(q) |
            filtered_df["AI_Summary"].str.lower().str.contains(q)
        ]
        
    # Display Count
    st.caption(f"Showing {len(filtered_df)} papers matching filters.")

    # --- List View Rendering ---
    # Pagination
    PAPERS_PER_PAGE = 20
    if 'page' not in st.session_state: st.session_state.page = 0
    
    total_pages = max(1, (len(filtered_df) - 1) // PAPERS_PER_PAGE + 1)
    
    # Page Controls Top
    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
    with col_p1:
        if st.button("⬅ Previous", key="prev_top"):
            st.session_state.page = max(0, st.session_state.page - 1)
            st.rerun()
    with col_p2:
        st.markdown(f"<div style='text-align: center'><b>Page {st.session_state.page + 1} / {total_pages}</b></div>", unsafe_allow_html=True)
    with col_p3:
        if st.button("Next ➡", key="next_top"):
            st.session_state.page = min(total_pages - 1, st.session_state.page + 1)
            st.rerun()

    # Slice Data
    start_idx = st.session_state.page * PAPERS_PER_PAGE
    end_idx = start_idx + PAPERS_PER_PAGE
    page_indices = filtered_df.index[start_idx:end_idx]
    
    # Render Items
    for idx in page_indices:
        row = df.loc[idx]
        status = row["Select? (Y/N)"]
        
        # Card Container
        with st.container():
            # Layout: [Check/Cross] | [Content]
            c_act, c_content = st.columns([1, 6])
            
            with c_act:
                # Status Indicator
                if status == "Y":
                    st.success("Included")
                elif status == "N":
                    st.error("Excluded")
                else:
                    st.warning("Pending")
                
                # Buttons
                if st.button("✅ Include", key=f"inc_{idx}", use_container_width=True):
                    df.at[idx, "Select? (Y/N)"] = "Y"
                    save_data(df)
                    st.rerun()
                
                if st.button("❌ Exclude", key=f"exc_{idx}", use_container_width=True):
                    df.at[idx, "Select? (Y/N)"] = "N"
                    save_data(df)
                    st.rerun()

            with c_content:
                # Title
                title_html = highlight_text(row['Title'], search_query)
                st.markdown(f"<h4 style='margin-top:0; margin-bottom:5px'>{title_html}</h4>", unsafe_allow_html=True)
                
                # Metadata Badges
                # Custom CSS for badges
                badge_style = "display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.8em; margin-right:5px; font-weight:500;"
                
                badges_html = f"""
                <div style="margin-bottom: 10px;">
                    <span style="{badge_style} background-color:#e2e8f0; color:#2d3748;">{row['Journal_Category']}</span>
                    <span style="{badge_style} background-color:#edf2f7; color:#4a5568;">{row['PubDate']}</span>
                """
                
                if row.get('Cancer_Type'):
                    badges_html += f'<span style="{badge_style} background-color:#fee2e2; color:#c53030;">🦠 {row["Cancer_Type"]}</span>'
                else:
                    badges_html += f'<span style="{badge_style} background-color:#e2e8f0; color:#4a5568;">❓ 未指定癌肿</span>'
                    
                if row.get('Study_Design'):
                    badges_html += f'<span style="{badge_style} background-color:#e0e7ff; color:#2c5282;">📊 {row["Study_Design"]}</span>'
                else:
                    badges_html += f'<span style="{badge_style} background-color:#edf2f7; color:#718096;">⚪ 未知设计</span>'
                    
                phase = row.get('Phase', '')
                if phase and phase != "nan":
                    if "III" in str(phase) or "3" in str(phase):
                        # Highlight Phase 3 in PURPLE
                        badges_html += f'<span style="{badge_style} background-color:#e9d8fd; color:#44337a; border: 1px solid #b794f4;">⚠️ {phase}</span>'
                    elif phase == "N/A" or phase == "NA":
                        # Explicitly show N/A in Gray
                        badges_html += f'<span style="{badge_style} background-color:#f7fafc; color:#a0aec0;">⚪ N/A</span>'
                    else:
                        # Normal Phase (Green)
                        badges_html += f'<span style="{badge_style} background-color:#f0fff4; color:#276749;">⚡ {phase}</span>'
                else:
                     badges_html += f'<span style="{badge_style} background-color:#f7fafc; color:#a0aec0;">⚪ N/A</span>'
                    
                badges_html += f'<span style="{badge_style} background-color:#f7fafc; color:#718096;">PMID: {row["PMID"]}</span>'
                badges_html += "</div>"
                
                st.markdown(badges_html, unsafe_allow_html=True)
                
                # AI Summary
                ai_sum = row.get('AI_Summary', '')
                if ai_sum:
                    ai_html = highlight_text(ai_sum, search_query)
                    st.info(f"**AI Summary**: {ai_html}")
                    
                # Highlights
                highlights = row.get('Highlights', '')
                if highlights:
                    # Format highlights (assuming semi-colon separated)
                    # Handle both Chinese and English semi-colons
                    points = re.split(r'[;；]', str(highlights))
                    points = [p.strip() for p in points if p.strip()]
                    
                    if points:
                        hl_html = "<div style='margin-bottom:10px; background-color:#fffaf0; padding:10px; border-radius:4px; border-left:3px solid #ed8936'>"
                        hl_html += "<div style='font-weight:bold; color:#c05621; margin-bottom:5px'>✨ Key Highlights</div>"
                        hl_html += "<ul style='margin:0; padding-left:20px; color:#744210'>"
                        for p in points:
                            hl_html += f"<li>{highlight_text(p, search_query)}</li>"
                        hl_html += "</ul></div>"
                        st.markdown(hl_html, unsafe_allow_html=True)
                
                # Abstract Expander
                with st.expander("📄 Abstract"):
                    # 1. Highlight keywords first
                    abs_text = highlight_text(row['Abstract'], search_query)
                    # 2. Then structure formatting
                    abs_text = format_abstract_structure(abs_text)
                    st.markdown(abs_text, unsafe_allow_html=True)
                    
            st.divider()

if __name__ == "__main__":
    main()
