# Sales Call Analysis Tool

This tool analyzes recorded outbound sales calls to provide objective reports with action items and training feedback for sales agents to improve their performance on future calls.

## Features

- **Call Transcription**: Supports audio transcription (note: requires additional setup for Gemini)
- **AI-Powered Analysis**: Uses Google's Gemini AI to analyze sales techniques and conversation quality
- **Comprehensive Reports**: Generates detailed reports with:
  - Call summary
  - Agent strengths
  - Areas for improvement
  - Specific action items
  - Sales techniques assessment
  - Customer engagement score
  - Overall performance score
- **User-Friendly Interface**: Easy-to-use Streamlit web application
- **Batch Processing**: Analyze multiple calls from a CSV file
- **Report Export**: Save analysis results as markdown files

## Requirements

- Python 3.8+
- Google API key with access to Gemini

## Installation

1. Clone this repository or navigate to the project directory
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Open your web browser and go to the URL displayed in the terminal (typically http://localhost:8501)

3. Enter your Google API key in the sidebar

4. Upload a CSV file containing call records with a "Recording URL" column

5. Click "Analyze Calls" to start the analysis process

6. View the results in the web interface and access saved reports in the "reports" directory

## CSV Format

The tool expects a CSV file with the following columns:
- `Recording URL` (required): URL to the audio recording of the call
- `Agent`: Name of the sales agent
- `Call Date`: Date of the call
- `Duration`: Duration of the call
- `Call Type`: Type of call (e.g., Manual, Inbound)
- `Disposition`: Call disposition (e.g., Answered, Did Not Pick up)
- `Call ID`: Unique identifier for the call

## Notes

- For demonstration purposes, the tool is limited to analyzing the first 5 calls with recordings to avoid excessive API costs
- A Google API key with access to Gemini is required for analysis
- Audio transcription with Gemini requires additional setup. The current implementation includes a placeholder for the transcription functionality.
- Audio files are temporarily downloaded for processing and then deleted
- Reports are saved in the "reports" directory in markdown format

## Troubleshooting

- If you encounter issues with audio downloads, check that the Recording URLs are accessible
- Ensure your Google API key has access to the Gemini API and sufficient quota
- For large CSV files, the analysis may take some time to complete
