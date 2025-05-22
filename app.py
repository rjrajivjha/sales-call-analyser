import os
import sys
import tempfile
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from call_analyzer import SalesCallAnalyzer
from datetime import datetime
import logging
from logging_config import setup_logger
from cache_utils import CACHE_DIR, get_call_hash, load_from_cache # For accessing cache

# Set up logging
logger = setup_logger()
logger.info("Starting application")

# Redirect stdout and stderr to logger
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        try:
            for line in buf.rstrip().splitlines():
                if line.strip():  # Only log non-empty lines
                    self.logger.log(self.log_level, line.rstrip())
        except Exception as e:
            self.logger.error(f"Error in StreamToLogger.write: {str(e)}", exc_info=True)

    def flush(self):
        pass

# Redirect stdout and stderr to logger
try:
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    logger.info("Successfully redirected stdout/stderr to logger")
except Exception as e:
    logger.error(f"Failed to redirect stdout/stderr: {str(e)}", exc_info=True)
    raise

st.set_page_config(
    page_title="Sales Call Analyzer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.2rem;
        color: #1976D2;
        font-weight: 600;
        margin-top: 1rem;
    }
    .report-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def extract_scores_from_markdown(md_file):
    """Extract scores from markdown file content."""
    try:
        with open(md_file, 'r') as f:
            content = f.read()
            engagement_match = re.search(r'\*\*Customer Engagement Score:\*\*\s*([^\n/]+)', content)
            performance_match = re.search(r'\*\*Overall Performance Score:\*\*\s*([^\n/]+)', content)
            return (
                engagement_match.group(1).strip() if engagement_match else 'N/A',
                performance_match.group(1).strip() if performance_match else 'N/A'
            )
    except Exception as e:
        logger.error(f"Error extracting scores from {md_file}: {str(e)}")
        return 'N/A', 'N/A'

def display_reports_page():
    st.markdown('<p class="main-header">Call Analysis Reports</p>', unsafe_allow_html=True)
    st.markdown("Browse and filter all generated call analysis reports.")

    # Get both JSON cache files and markdown report files
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    report_files = [f for f in os.listdir('reports') if f.endswith('.md')]
    all_reports_data = []

    if not cache_files and not report_files:
        st.info("No reports found. Analyze some calls to generate reports.")
        return

    # Process JSON cache files
    for cache_file in cache_files:
        try:
            file_path = os.path.join(CACHE_DIR, cache_file)
            with open(file_path, 'r') as f:
                cached_data = json.load(f)
                report = cached_data.get('result')
                if report and report.get('success'):
                    call_data = report.get('call_data', {})
                    all_reports_data.append({
                        'Call ID': call_data.get('Call ID', 'N/A'),
                        'Agent': call_data.get('Agent', 'N/A'),
                        'Call Date': call_data.get('Call Date', 'N/A'),
                        'Duration': call_data.get('Duration', 'N/A'),
                        'Engagement Score': report.get('engagement_score', 'N/A'),
                        'Performance Score': report.get('performance_score', 'N/A'),
                        '_source': 'cache',
                        '_file_path': file_path
                    })
        except Exception as e:
            logger.error(f"Error loading cached report {cache_file}: {str(e)}", exc_info=True)
    
    # Process markdown report files
    for report_file in report_files:
        try:
            # Skip if we already have this report from cache
            call_id = report_file.split('_')[-2]  # Extract call ID from filename
            if not any(r.get('Call ID') == call_id for r in all_reports_data):
                # Extract scores from markdown
                engagement, performance = extract_scores_from_markdown(os.path.join('reports', report_file))
                if engagement != 'N/A' or performance != 'N/A':
                    all_reports_data.append({
                        'Call ID': call_id,
                        'Agent': report_file.split('_')[2].replace('_', ' '),  # Extract agent name
                        'Call Date': 'N/A',
                        'Duration': 'N/A',
                        'Engagement Score': engagement,
                        'Performance Score': performance,
                        '_source': 'markdown',
                        '_file_path': os.path.join('reports', report_file)
                    })
        except Exception as e:
            logger.error(f"Error processing markdown report {report_file}: {str(e)}", exc_info=True)

    if not all_reports_data:
        st.info("No valid reports found in cache.")
        return

    df_reports = pd.DataFrame(all_reports_data)
    
    # Make table filterable - basic version, can be enhanced with st-aggrid or similar
    st.sidebar.header("Filter Reports")
    filter_agent = st.sidebar.multiselect("Filter by Agent", options=df_reports['Agent'].unique(), default=df_reports['Agent'].unique())
    filter_call_id = st.sidebar.text_input("Filter by Call ID (contains)")

    filtered_df = df_reports.copy()
    if filter_agent:
        filtered_df = filtered_df[filtered_df['Agent'].isin(filter_agent)]
    if filter_call_id:
        filtered_df = filtered_df[filtered_df['Call ID'].str.contains(filter_call_id, case=False, na=False)]

    # Display the filtered reports table without the internal columns
    display_columns = ['Call ID', 'Agent', 'Call Date', 'Duration', 'Engagement Score', 'Performance Score']
    st.dataframe(filtered_df[display_columns])

    # Detailed report viewer
    st.markdown("---")
    st.markdown("<p class='sub-header'>View Detailed Report</p>", unsafe_allow_html=True)
    
    if not filtered_df.empty:
        report_to_view_id = st.selectbox(
            "Select Call ID to view full report:", 
            options=filtered_df['Call ID'].tolist()
        )
        
        if report_to_view_id:
            selected_report = filtered_df[filtered_df['Call ID'] == report_to_view_id].iloc[0]
            
            # For cached reports, load the full report data
            if selected_report['_source'] == 'cache':
                with open(selected_report['_file_path'], 'r') as f:
                    report_data = json.load(f)['result']
            # For markdown reports, create a simplified report structure
            else:
                with open(selected_report['_file_path'], 'r') as f:
                    content = f.read()
                    # Extract transcript and analysis sections if they exist
                    transcript_match = re.search(r'## Call Transcript\n\n([\s\S]*?)(?=\n##|$)', content)
                    analysis_match = re.search(r'## Analysis\n\n([\s\S]*)', content)
                    
                    report_data = {
                        'call_data': {
                            'Call ID': selected_report['Call ID'],
                            'Agent': selected_report['Agent'],
                            'Call Date': selected_report['Call Date'],
                            'Duration': selected_report['Duration']
                        },
                        'transcript': transcript_match.group(1).strip() if transcript_match else 'No transcript available',
                        'analysis': analysis_match.group(1).strip() if analysis_match else 'No analysis available',
                        'engagement_score': selected_report['Engagement Score'].split('/')[0] if '/' in str(selected_report['Engagement Score']) else 'N/A',
                        'performance_score': selected_report['Performance Score'].split('/')[0] if '/' in str(selected_report['Performance Score']) else 'N/A',
                        'success': True
                    }
            
            if report_data:
                st.markdown(f"### Detailed Report for Call ID: {report_data.get('call_data', {}).get('Call ID', 'N/A')}")
                display_single_report_details(report_data)
    else:
        st.info("No reports match the current filters.")


def display_single_report_details(report_result):
    """Displays the details of a single report (used by modal/detail view)."""
    st.markdown(f"**Agent:** {report_result['call_data'].get('Agent', 'N/A')}")
    st.markdown(f"**Call Date:** {report_result['call_data'].get('Call Date', 'N/A')}")
    st.markdown(f"**Duration:** {report_result['call_data'].get('Duration', 'N/A')}")
    
    st.markdown("#### Transcript")
    st.markdown(
        f'<div style="white-space: pre-wrap; background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">'
        f'{report_result.get("transcript", "No transcript available.")}'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("#### Full Analysis")
    st.markdown(report_result.get('analysis', 'No analysis available.'))
    st.markdown(f"**Customer Engagement Score:** {report_result.get('engagement_score', 'N/A')}/10")
    st.markdown(f"**Overall Performance Score:** {report_result.get('performance_score', 'N/A')}/10")


def main_analyzer_page():
    logger.info("Displaying Main Analyzer Page")
    # Header
    st.markdown('<p class="main-header">Sales Call Analyzer</p>', unsafe_allow_html=True)
    st.markdown("""
    This tool analyzes recorded sales calls to provide objective feedback, 
    action items, and training recommendations for sales agents.
    """)
    
    # File upload
    st.markdown('<p class="sub-header">Upload Call Data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file with call records", type=["csv"])
    
    if uploaded_file is not None:
        try:
            logger.info(f"CSV file uploaded: {uploaded_file.name} (size: {len(uploaded_file.getvalue())} bytes)")
            
            # Save the uploaded file temporarily
            temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Read the CSV file
            df = pd.read_csv(temp_file_path)
            
            # Count calls with recordings
            recording_count = df['Recording URL'].notna().sum()
            st.info(f"Found {recording_count} calls with recordings out of {len(df)} total calls.")
            
            # Analyze button
            if st.button("Analyze Calls"):
                api_key_present = os.getenv("GOOGLE_API_KEY")
                logger.info("Analyze button clicked")
                if not api_key_present:
                    error_msg = "No Google API key found in environment. Please set it in the sidebar or .env file."
                    logger.error(error_msg)
                    st.error(error_msg)
                elif recording_count == 0:
                    error_msg = "No calls with recordings found in the CSV"
                    logger.error(error_msg)
                    st.error("No calls with recordings found in the CSV.")
                else:
                    logger.info(f"Starting analysis of up to 2 longest calls from {recording_count} calls with recordings")
                    # Initialize progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Filter for rows with recording URLs and convert handling time to seconds for sorting
                    df_with_recordings = df[df['Recording URL'].notna() & (df['Recording URL'] != '')].copy()
                    
                    # Convert handling time string (HH:MM:SS) to seconds for sorting
                    def duration_to_seconds(time_str):
                        try:
                            h, m, s = map(int, time_str.split(':'))
                            return h * 3600 + m * 60 + s
                        except:
                            return 0 # or handle error appropriately
                            
                    df_with_recordings['duration_seconds'] = df_with_recordings['Duration'].apply(duration_to_seconds)
                    
                    st.info(f"Total duration of calls with recordings: {df_with_recordings['duration_seconds'].sum()} seconds")

                    # Sort by duration in descending order and take top 5 longest calls
                    df_with_recordings = df_with_recordings.sort_values('duration_seconds', ascending=False).head(5)
                    
                    st.info(f"Selected {len(df_with_recordings)} calls with longest duration for analysis (Top 5) {df_with_recordings['duration_seconds'].sum()} seconds")
                    
                    # Initialize analyzer
                    analyzer = SalesCallAnalyzer()
                    
                    # Process each call
                    results = []
                    for i, (_, row) in enumerate(df_with_recordings.iterrows()):
                        call_id_display = row.get('Call ID', f'Row {i+1}')
                        agent_display = row.get('Agent', 'Unknown')
                        logger.info(f"Processing call {i+1}/{len(df_with_recordings)} - ID: {call_id_display}, Agent: {agent_display}")
                        
                        # Update progress
                        progress = (i + 1) / len(df_with_recordings)
                        progress_bar.progress(progress)
                        status_text.text(f"Analyzing call {i + 1} of {len(df_with_recordings)}: {call_id_display}")
                        
                        try:
                            result = analyzer.process_call(row)
                            results.append(result)
                            if result.get('success'):
                                logger.info(f"Successfully analyzed call {call_id_display}")
                            else:
                                logger.warning(f"Analysis failed for call {call_id_display}: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            logger.error(f"Error processing call {call_id_display}: {str(e)}", exc_info=True)
                            results.append({
                                'success': False,
                                'error': f"Error processing call: {str(e)}",
                                'call_data': row.to_dict()
                            })
                        
                        progress_bar.progress((i+1) / len(df_with_recordings))
                    
                    status_text.text("Analysis complete!")
                    progress_bar.progress(1.0)
                    
                    # Display results
                    display_results(results)
                    
                    # Save results
                    logger.info(f"Analysis complete. Successfully processed {len([r for r in results if r.get('success')])}/{len(results)} calls")
                    save_results(results)
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
            st.error("An error occurred while processing the uploaded file. Please check the logs for more details.")

def main():
    logger.info("Starting Sales Call Analyzer application")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page_options = ["Call Analyzer", "View Reports"]
    selected_page = st.sidebar.radio("Go to", page_options)

    # API Key input in sidebar (common for all pages)
    st.sidebar.markdown('<p class="sub-header">Settings</p>', unsafe_allow_html=True)
    api_key = st.sidebar.text_input("Google API Key", type="password", key="sidebar_api_key")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        logger.info("Google API Key set from sidebar.")
    elif not os.getenv("GOOGLE_API_KEY"):
        logger.warning("Google API Key not set. Please enter it in the sidebar or set GOOGLE_API_KEY environment variable.")
        # Optionally, show a warning on the main page if key is missing and trying to analyze

    st.sidebar.markdown("---")
    st.sidebar.markdown("<p class='section-header'>Instructions</p>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    **Call Analyzer Page:**
    1. Upload a CSV file with call records.
    2. Ensure your Google API key is entered in Settings.
    3. Click 'Analyze Calls' to start.
    
    **View Reports Page:**
    - Browse and filter all generated reports.
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("<p class='section-header'>About</p>", unsafe_allow_html=True)
    st.sidebar.markdown("Developed for Chakr Innovations")

    if selected_page == "Call Analyzer":
        main_analyzer_page()
    elif selected_page == "View Reports":
        display_reports_page()
    
    # The main analyzer page is now handled by main_analyzer_page()
    # This prevents duplicate file uploader widgets
    pass

def display_results(results):
    """Display the analysis results in the Streamlit app."""
    st.markdown('<p class="sub-header">Analysis Results</p>', unsafe_allow_html=True)
    
    # Count successful analyses
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    # Display metrics in a container
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Total Calls Analyzed</p>
        </div>
        '''.format(len(results)), unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Successfully Analyzed</p>
        </div>
        '''.format(successful), unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Failed Analyses</p>
        </div>
        '''.format(failed), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display individual reports
    for i, result in enumerate(results):
        if result.get('success', False):
            with st.expander(f"Call Report: {result['call_data'].get('Agent', 'Unknown')} - {result['call_data'].get('Call Date', 'Unknown')}"):
                st.markdown('<div class="report-container">', unsafe_allow_html=True)
                
                # Call details
                st.markdown('<p class="section-header">Call Details</p>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Agent:** {result['call_data'].get('Agent', 'Unknown')}")
                    st.markdown(f"**Date:** {result['call_data'].get('Call Date', 'Unknown')}")
                    st.markdown(f"**Customer Engagement Score:** {result.get('engagement_score', 'N/A')}/10")
                with col2:
                    st.markdown(f"**Duration:** {result['call_data'].get('Duration', 'Unknown')}")
                    st.markdown(f"**Call Type:** {result['call_data'].get('Call Type', 'Unknown')}")
                    st.markdown(f"**Overall Performance Score:** {result.get('performance_score', 'N/A')}/10")
                with col3:
                    st.markdown(f"**Disposition:** {result['call_data'].get('Disposition', 'Unknown')}")
                    st.markdown(f"**Call ID:** {result['call_data'].get('Call ID', 'Unknown')}")
                
                # Transcript and Analysis in tabs
                tab1, tab2 = st.tabs(["Call Transcript", "Analysis"])
                
                with tab1:
                    st.markdown(result['transcript'])
                
                with tab2:
                    st.markdown(result['analysis'])
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.expander(f"Failed Analysis: {result['call_data'].get('Agent', 'Unknown') if 'call_data' in result else 'Unknown'} - {result['call_data'].get('Call Date', 'Unknown') if 'call_data' in result else 'Unknown'}"):
                st.error(f"Error: {result.get('error', 'Unknown error')}")

def save_results(results):
    """Save the analysis results to files."""
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a directory for reports if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Save each successful analysis as a separate markdown file
    for i, result in enumerate(results):
        if result.get('success', False):
            agent_name = result['call_data'].get('Agent', 'Unknown').replace(' ', '_')
            call_date = result['call_data'].get('Call Date', 'Unknown').replace('-', '')
            call_id = result['call_data'].get('Call ID', 'Unknown')
            
            filename = f"reports/call_analysis_{agent_name}_{call_date}_{call_id}_{timestamp}.md"
            
            with open(filename, 'w') as f:
                # Write call details
                f.write(f"# Call Analysis Report\n\n")
                f.write(f"## Call Details\n\n")
                f.write(f"- **Agent:** {result['call_data'].get('Agent', 'Unknown')}\n")
                f.write(f"- **Date:** {result['call_data'].get('Call Date', 'Unknown')}\n")
                f.write(f"- **Duration:** {result['call_data'].get('Duration', 'Unknown')}\n")
                f.write(f"- **Call Type:** {result['call_data'].get('Call Type', 'Unknown')}\n")
                f.write(f"- **Disposition:** {result['call_data'].get('Disposition', 'Unknown')}\n")
                f.write(f"- **Call ID:** {result['call_data'].get('Call ID', 'Unknown')}\n")
                f.write(f"- **Customer Engagement Score:** {result.get('engagement_score', 'N/A')}/10\n")
                f.write(f"- **Overall Performance Score:** {result.get('performance_score', 'N/A')}/10\n\n")
                
                # Write transcript
                f.write(f"## Call Transcript\n\n")
                f.write(f"{result['transcript']}\n\n")
                
                # Write analysis
                f.write(f"## Analysis\n\n")
                f.write(f"{result['analysis']}\n")
    
    # Save a summary of all analyses
    summary_filename = f"reports/summary_{timestamp}.md"
    with open(summary_filename, 'w') as f:
        f.write(f"# Call Analysis Summary\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Overview\n\n")
        f.write(f"- Total calls analyzed: {len(results)}\n")
        f.write(f"- Successfully analyzed: {sum(1 for r in results if r.get('success', False))}\n")
        f.write(f"- Failed analyses: {sum(1 for r in results if not r.get('success', False))}\n\n")
        
        f.write(f"## Call List\n\n")
        for i, result in enumerate(results):
            status = "✅ Success" if result.get('success', False) else "❌ Failed"
            if 'call_data' in result:
                agent = result['call_data'].get('Agent', 'Unknown')
                date = result['call_data'].get('Call Date', 'Unknown')
                call_id = result['call_data'].get('Call ID', 'Unknown')
                f.write(f"{i+1}. {status} - Agent: {agent}, Date: {date}, Call ID: {call_id}\n")
            else:
                f.write(f"{i+1}. {status} - Unknown call\n")
    
    st.success(f"Reports saved to the 'reports' directory.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in application: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please check the logs for more details.")
        raise
