# Oncology Literature Screening Tool

A Streamlit-based tool for screening 203 non-RCT oncology papers from top medical journals (2020-2026).

## Features
- **AI-Augmented**: Auto-generated summaries, cancer type classification, and highlights.
- **Smart Filters**: Filter by Cancer Type, Study Design, Phase, and Screening Status.
- **Visual Tags**: Color-coded badges for quick identification.
- **Structured Abstracts**: Automatically formatted abstracts for better readability.

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app_v2.py
   ```

3. Open your browser at `http://localhost:8501`.

## How to Deploy (Streamlit Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set the main file path to `app_v2.py`.
5. Click **Deploy**.

## Data Source
- Data is stored in `Literature_Screening_List.xlsx`.
- The app reads and writes directly to this file.
- **Note for Cloud Deployment**: Changes made on Streamlit Cloud are *ephemeral* (they will reset when the app restarts) unless you connect it to a database or Google Sheets. For simple viewing and filtering, it works perfectly. For collaborative tagging, consider downloading the CSV periodically.
