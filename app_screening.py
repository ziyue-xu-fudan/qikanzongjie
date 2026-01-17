import streamlit as st
import pandas as pd
import os
import re

# Page Config
st.set_page_config(page_title="Medical Literature Screener", layout="wide", initial_sidebar_state="expanded")

# File Path
FILE_PATH = "Literature_Screening_List.xlsx"

# --- Custom CSS for PubMed Style ---
st.markdown("""
<style>
    /* Global Font */
    body {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    
    /* Title */
    .paper-title {
        font-family: "Georgia", serif;
        color: #2b6cb0;
        font-size: 26px;
        font-weight: 600;
        line-height: 1.4;
        margin-bottom: 15px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 500;
        margin-right: 8px;
    }
    .badge-journal { background-color: #e2e8f0; color: #2d3748; }
    .badge-date { background-color: #edf2f7; color: #4a5568; }
    .badge-doi { background-color: #ebf8ff; color: #3182ce; text-decoration: none; }
    
    /* Abstract Sections */
    .abstract-section {
        margin-top: 12px;
        margin-bottom: 4px;
        text-transform: uppercase;
        font-weight: 700;
        font-size: 14px;
        color: #2d3748;
        letter-spacing: 0.5px;
    }
    .abstract-text {
        font-size: 16px;
        line-height: 1.6;
        color: #1a202c;
        margin-bottom: 12px;
    }
    
    /* Highlight */
    .highlight-kw {
        background-color: #fff3cd;
        padding: 0 2px;
        border-radius: 2px;
        border-bottom: 2px solid #ffc107;
    }
    
    /* AI Box */
    .ai-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3182ce;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 15px;
        color: #2c5282;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def load_data():
    if not os.path.exists(FILE_PATH):
        st.error(f"File not found: {FILE_PATH}")
        return None
    df = pd.read_excel(FILE_PATH, dtype=str)
    df = df.fillna("")
    # Convert PubDate to datetime for sorting/filtering if possible
    try:
        df['PubDate_DT'] = pd.to_datetime(df['PubDate'], errors='coerce')
        df['Year'] = df['PubDate_DT'].dt.year
    except:
        df['Year'] = 0
    return df

def save_data(df):
    # Drop temp columns before saving
    save_df = df.drop(columns=['PubDate_DT', 'Year'], errors='ignore')
    save_df.to_excel(FILE_PATH, index=False)

def format_abstract(text, search_kw=None):
    if not isinstance(text, str) or not text:
        return "<p class='abstract-text'>No abstract available.</p>"
    
    # 0. Highlight User Search Keyword (Highest Priority)
    if search_kw:
        # Using a distinct style for user search terms (e.g., orange background)
        pattern = re.compile(f"({re.escape(search_kw)})", re.IGNORECASE)
        text = pattern.sub(r'<span style="background-color: #ffccbc; border-bottom: 2px solid #e53e3e; font-weight: bold; padding: 0 2px;">\1</span>', text)

    # 1. Highlight Standard Keywords
    keywords = [
        "cohort", "prospective", "retrospective", "observational", 
        "registry", "real-world", "database", "population-based",
        "longitudinal", "case-control", "cross-sectional",
        "surveillance", "epidemiology", "incidence", "prevalence",
        "randomized", "trial", "efficacy", "safety"
    ]
    for word in keywords:
        # Avoid double highlighting if already matched by user keyword (simple check)
        # A more robust way is complex, but for now standard overlay is okay
        pattern = re.compile(f"({word})", re.IGNORECASE)
        text = pattern.sub(r'<span class="highlight-kw">\1</span>', text)

    # 2. Detect and Format Sections (PubMed Style)
    # Common section headers in UPPERCASE or Title Case followed by colon
    sections = [
        "BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS", "FINDINGS", "INTERPRETATION", 
        "OBJECTIVE", "DESIGN", "SETTING", "PARTICIPANTS", "MAIN OUTCOMES AND MEASURES",
        "IMPORTANCE", "Funding"
    ]
    
    formatted_html = ""
    
    # Simple heuristic: split by section headers
    # We construct a regex pattern that matches these headers
    pattern_str = "|".join([re.escape(s) for s in sections])
    # Look for Section Header followed by : or space at start of line or sentence
    # This is a bit tricky, let's try a simpler approach: replace known headers with HTML
    
    for section in sections:
        # Match "SECTION: " or "SECTION " (case insensitive, but prefer upper)
        # We replace it with <div class='abstract-section'>SECTION</div>
        # Use a lookahead to ensure we don't break mid-word
        regex = re.compile(f"(^|\\.\\s|\\n)({section})[:\\s]", re.IGNORECASE)
        
        def replace_func(match):
            prefix = match.group(1) # The period or newline before
            header = match.group(2).upper() # Normalize to UPPER
            return f"{prefix}<div class='abstract-section'>{header}</div>"
            
        text = regex.sub(replace_func, text)
        
    # Wrap the rest in paragraph tags, treating newlines as breaks if they aren't section headers
    # Actually, since we injected divs, we should just wrap text nodes.
    # Simpler: Just convert newlines to <br> if they are double
    text = text.replace("\n\n", "<br><br>")
    
    return f"<div class='abstract-text'>{text}</div>"

# --- Main App ---

def main():
    # Initialize Session State
    if 'df' not in st.session_state:
        st.session_state.df = load_data()
        
    if st.session_state.df is None:
        return

    df = st.session_state.df
    
    # --- Sidebar: Navigation ---
    with st.sidebar:
        st.header("🧭 Navigation")
        # Add "Batch Screening (List)" mode
        page_mode = st.radio("Go to:", ["Screening Mode (Single)", "Batch Screening (List)", "Included Papers (Results)"])
        st.divider()

    if page_mode == "Included Papers (Results)":
        # ... (Previous Included Papers code) ...
        st.title("📂 Included Papers")
        
        # Filter only included
        included_df = df[df["Select? (Y/N)"] == "Y"].copy()
        
        st.metric("Total Included", len(included_df))
        
        if len(included_df) > 0:
            # 1. Export Excel
            st.subheader("1. Export List")
            # Convert to CSV for download
            csv = included_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Excel (CSV)",
                csv,
                "Included_Papers.csv",
                "text/csv",
                key='download-csv'
            )
            
            # 2. Export DOIs
            st.subheader("2. DOI List (for Sci-Hub/Zotero)")
            doi_list = "\n".join([str(doi) for doi in included_df['DOI'] if str(doi) and str(doi) != "nan"])
            st.text_area("Copy these DOIs:", doi_list, height=150)
            
            # 3. Preview Table
            st.subheader("3. Preview")
            st.dataframe(
                included_df[["Title", "Journal_Category", "PubDate", "AI_Summary"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No papers marked as 'Y' yet. Go back to Screening Mode!")
        return

    # --- Sidebar: Filters (PubMed Style) ---
    with st.sidebar:
        st.header("🔍 Filters")
        
        # 1. Progress
        total = len(df)
        reviewed = len(df[df["Select? (Y/N)"].isin(["Y", "N"])])
        st.progress(reviewed / total if total > 0 else 0)
        st.caption(f"Reviewed: {reviewed} / {total} ({reviewed/total:.1%})")
        
        st.divider()
        
        # 2. Status Filter
        filter_status = st.radio(
            "Screening Status", 
            ["Unreviewed", "All Papers", "Included", "Excluded"],
            index=0
        )
        
        # 3. Journal Filter (Multi-select)
        all_journals = sorted(df["Journal_Category"].unique().tolist())
        selected_journals = st.multiselect("Journal", all_journals, default=all_journals)
        
        # 4. Year Filter
        min_year = int(df['Year'].min()) if df['Year'].min() > 0 else 2023
        max_year = int(df['Year'].max()) if df['Year'].max() > 0 else 2026
        selected_years = st.slider("Publication Year", min_year, max_year, (min_year, max_year))
        
        # 5. Text Search
        search_query = st.text_input("Search Keywords", placeholder="e.g. lung cancer, cohort")
        
        st.divider()
        if st.button("💾 Save & Reload"):
            save_data(df)
            st.session_state.df = load_data()
            st.rerun()

    # --- Filtering Logic ---
    filtered_df = df.copy()
    
    # Status
    if filter_status == "Unreviewed":
        filtered_df = filtered_df[~filtered_df["Select? (Y/N)"].isin(["Y", "N"])]
    elif filter_status == "Included":
        filtered_df = filtered_df[filtered_df["Select? (Y/N)"] == "Y"]
    elif filter_status == "Excluded":
        filtered_df = filtered_df[filtered_df["Select? (Y/N)"] == "N"]
        
    # Journal
    if selected_journals:
        filtered_df = filtered_df[filtered_df["Journal_Category"].isin(selected_journals)]
        
    # Year
    filtered_df = filtered_df[(filtered_df['Year'] >= selected_years[0]) & (filtered_df['Year'] <= selected_years[1])]
    
    # Search
    if search_query:
        q = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["Title"].str.lower().str.contains(q) | 
            filtered_df["Abstract"].str.lower().str.contains(q) |
            filtered_df["AI_Summary"].str.lower().str.contains(q)
        ]
    
    filtered_indices = filtered_df.index.tolist()

    if not filtered_indices:
        st.success("✨ No papers match your filters. Great job!")
        return

    # === BATCH SCREENING MODE ===
    if page_mode == "Batch Screening (List)":
        st.title("📚 Batch Screening")
        st.caption(f"Showing {len(filtered_indices)} papers")
        
        # Pagination
        PAPERS_PER_PAGE = 20
        if 'batch_page' not in st.session_state:
            st.session_state.batch_page = 0
            
        total_pages = (len(filtered_indices) - 1) // PAPERS_PER_PAGE + 1
        
        # Page controls
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            if st.button("⬅️ Previous", disabled=(st.session_state.batch_page == 0)):
                st.session_state.batch_page -= 1
                st.rerun()
        with c2:
            st.markdown(f"<div style='text-align: center'>Page {st.session_state.batch_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
        with c3:
            if st.button("Next ➡️", disabled=(st.session_state.batch_page >= total_pages - 1)):
                st.session_state.batch_page += 1
                st.rerun()
                
        start_idx = st.session_state.batch_page * PAPERS_PER_PAGE
        end_idx = min(start_idx + PAPERS_PER_PAGE, len(filtered_indices))
        
        current_batch_indices = filtered_indices[start_idx:end_idx]
        
        # Render List
        for i, idx in enumerate(current_batch_indices):
            row = df.loc[idx]
            status = row["Select? (Y/N)"]
            status_color = "green" if status == "Y" else "red" if status == "N" else "gray"
            status_icon = "✅" if status == "Y" else "❌" if status == "N" else "⬜"
            
            with st.container():
                # Header Line
                c_chk, c_info, c_act = st.columns([0.5, 8, 2.5])
                
                with c_info:
                    st.markdown(f"**{i + start_idx + 1}. {row['Title']}**")
                    st.caption(f"{row['Journal_Category']} | {row['PubDate']} | Status: {status_icon}")
                    
                    # AI Summary Preview
                    ai_sum = str(row.get('AI_Summary', ''))
                    if ai_sum and ai_sum != "nan":
                        st.markdown(f"<span style='color: #2b6cb0; font-size: 0.9em'>🤖 {ai_sum}</span>", unsafe_allow_html=True)
                
                with c_act:
                    # Inline Actions
                    c_inc, c_exc = st.columns(2)
                    if c_inc.button("✔", key=f"b_inc_{idx}", help="Include"):
                        df.at[idx, "Select? (Y/N)"] = "Y"
                        save_data(df)
                        st.rerun()
                    if c_exc.button("✖", key=f"b_exc_{idx}", help="Exclude"):
                        df.at[idx, "Select? (Y/N)"] = "N"
                        save_data(df)
                        st.rerun()
                
                # Expander for details
                with st.expander("View Abstract"):
                    st.markdown(format_abstract(row['Abstract'], search_kw=search_query), unsafe_allow_html=True)
                    notes = st.text_input("Notes:", value=row.get("Reason / Notes", ""), key=f"b_note_{idx}")
                    if notes != row.get("Reason / Notes", ""):
                        df.at[idx, "Reason / Notes"] = notes
                        save_data(df)
                
                st.divider()
        return

    # === SCREENING MODE (Single View) ===
    # ... (Previous Single View code) ...
    # --- Navigation Logic ---
    if 'pointer' not in st.session_state:
        st.session_state.pointer = 0
    
    # Reset pointer if out of bounds
    if st.session_state.pointer >= len(filtered_indices):
        st.session_state.pointer = 0
        
    current_idx = filtered_indices[st.session_state.pointer]
    row = df.loc[current_idx]

    # --- Main Content Area ---
    
    # 1. Metadata Badge Row
    col_meta, col_nav = st.columns([3, 1])
    with col_meta:
        st.markdown(f"""
            <span class="badge badge-journal">{row['Journal_Category']}</span>
            <span class="badge badge-date">{row['PubDate']}</span>
            <a href="https://doi.org/{row['DOI']}" target="_blank" class="badge badge-doi">DOI Link ↗</a>
        """, unsafe_allow_html=True)
    
    with col_nav:
        st.caption(f"Paper {st.session_state.pointer + 1} of {len(filtered_indices)}")

    # 2. Title
    title_html = row["Title"]
    if search_query:
        pattern = re.compile(f"({re.escape(search_query)})", re.IGNORECASE)
        title_html = pattern.sub(r'<span style="background-color: #ffccbc; padding: 0 2px;">\1</span>', title_html)
    st.markdown(f'<div class="paper-title">{title_html}</div>', unsafe_allow_html=True)
    
    # 3. AI Summary Box
    ai_summary = str(row.get('AI_Summary', ''))
    if ai_summary and ai_summary != "nan":
        if search_query:
            pattern = re.compile(f"({re.escape(search_query)})", re.IGNORECASE)
            ai_summary = pattern.sub(r'<span style="background-color: #ffccbc; padding: 0 2px;">\1</span>', ai_summary)
            
        st.markdown(f"""
        <div class="ai-box">
            <b>🤖 AI 核心总结：</b><br>
            {ai_summary}
        </div>
        """, unsafe_allow_html=True)
        
    # 4. Structured Abstract
    st.markdown(format_abstract(row['Abstract'], search_kw=search_query), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 5. Action Bar (Sticky-like)
    c1, c2, c3, c4 = st.columns([1.5, 1.5, 3, 1])
    
    with c1:
        if st.button("✅ INCLUDE", type="primary", use_container_width=True, key=f"inc_{current_idx}"):
            df.at[current_idx, "Select? (Y/N)"] = "Y"
            save_data(df)
            st.rerun() # Pointer stays same, but list shrinks if 'Unreviewed' mode
            
    with c2:
        if st.button("❌ EXCLUDE", use_container_width=True, key=f"exc_{current_idx}"):
            df.at[current_idx, "Select? (Y/N)"] = "N"
            save_data(df)
            st.rerun()
            
    with c3:
        # Auto-save notes on change
        new_note = st.text_input("📝 Notes", value=row.get("Reason / Notes", ""), 
                               placeholder="Why included/excluded?", label_visibility="collapsed",
                               key=f"note_{current_idx}")
        if new_note != row.get("Reason / Notes", ""):
            df.at[current_idx, "Reason / Notes"] = new_note
            save_data(df)

    with c4:
        if st.button("Skip ⏭️", use_container_width=True):
            st.session_state.pointer = (st.session_state.pointer + 1) % len(filtered_indices)
            st.rerun()

if __name__ == "__main__":
    main()
