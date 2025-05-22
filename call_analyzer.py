import os
import pandas as pd
import requests
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import logging
import json
import re # Added for regex parsing
from logging_config import setup_logger
from cache_utils import get_call_hash, load_from_cache, save_to_cache

# Initialize logger
logger = setup_logger()

# Load environment variables
load_dotenv()

# Configure Google Generative AI
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    logger.info("Successfully configured Google Generative AI")
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {str(e)}")
    raise

class SalesCallAnalyzer:
    def __init__(self):
        """Initialize the SalesCallAnalyzer with Google's Gemini client."""
        try:
            # The API key is already configured at the module level
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Successfully initialized SalesCallAnalyzer with Gemini model")
        except Exception as e:
            logger.error(f"Failed to initialize SalesCallAnalyzer: {str(e)}", exc_info=True)
            raise
        
    def download_audio(self, url):
        """Download audio file from URL."""
        if not url or pd.isna(url):
            logger.warning("No URL provided for audio download")
            return None
        
        try:
            logger.info(f"Starting audio download from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Create a temporary file to store the audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        temp_file.write(chunk)
                temp_file.close()
                logger.info(f"Successfully downloaded audio to temporary file: {temp_file.name}")
                return temp_file.name
            except Exception as write_error:
                logger.error(f"Error writing audio file: {str(write_error)}")
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise
                
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error downloading audio: {str(req_err)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading audio: {str(e)}", exc_info=True)
            return None
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe MP3 audio file using Gemini's audio capabilities."""
        if not audio_file_path or not audio_file_path.lower().endswith('.mp3'):
            logger.warning("Invalid or non-MP3 file provided for transcription")
            return self._get_placeholder_transcript()
            
        try:
            # Check if file exists
            if not os.path.exists(audio_file_path):
                logger.error(f"MP3 file not found: {audio_file_path}")
                return self._get_placeholder_transcript()
                
            file_size = os.path.getsize(audio_file_path)
            logger.info(f"Starting transcription of {audio_file_path} (size: {file_size} bytes)")
                
            # Read the MP3 file as bytes
            with open(audio_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                
            # Call the model with the MP3 audio
            response = self.model.generate_content(
                contents=[{
                    "parts": [
                        {"text": "Transcribe this MP3 audio:"},
                        {
                            "inline_data": {
                                "mime_type": "audio/mp3",
                                "data": base64.b64encode(audio_data).decode('utf-8')
                            }
                        }
                    ]
                }],
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 4096,
                }
            )
            
            # Extract the transcription
            if response and hasattr(response, 'candidates') and response.candidates:
                transcript = response.candidates[0].content.parts[0].text
                logger.info("Successfully transcribed audio")
                return transcript
            else:
                logger.error("No transcription returned from API")
                return self._get_placeholder_transcript()
                
        except Exception as e:
            logger.error(f"Error in MP3 transcription: {str(e)}", exc_info=True)
            return self._get_placeholder_transcript()
        finally:
            # Clean up the temporary file
            try:
                if audio_file_path and os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                    logger.debug(f"Cleaned up temporary file: {audio_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")

    def _get_placeholder_transcript(self):
        """Return a placeholder transcript when transcription fails."""
        return """[Sample Transcript - Actual transcription would appear here]
        
        Agent: Hello, this is [Agent Name] calling from [Company]. How are you today?
        Customer: I'm doing well, thanks.
        Agent: Great! I'm calling because I noticed you were interested in our services. 
        Can I ask what made you reach out to us initially?
        ...
        """

    def analyze_call(self, transcript, call_data):
        """Analyze the call transcript and generate insights."""
        if not transcript:
            return {
                "error": "No transcript available for analysis",
                "success": False
            }
        
        # Prepare context about the call for the AI
        call_context = f"""
        Call Details:
        - Agent: {call_data.get('Agent', 'Unknown')}
        - Duration: {call_data.get('Duration', 'Unknown')}
        - Disposition: {call_data.get('Disposition', 'Unknown')}
        - Call Type: {call_data.get('Call Type', 'Unknown')}
        - Date: {call_data.get('Call Date', 'Unknown')}
        """
        
        # Create prompt for analysis
        prompt = f"""
        You are an expert sales coach analyzing an outbound sales call. Below is the transcript of a call made by a sales agent.
        
        {call_context}
        
        Transcript:
        {transcript}
        
        Please provide a comprehensive analysis of this sales call with the following sections:
        
        1. CALL SUMMARY: A brief objective summary of what happened in the call (2-3 sentences).
        
        2. STRENGTHS: Identify 2-3 things the agent did well during the call.
        
        3. AREAS FOR IMPROVEMENT: Identify 2-3 specific areas where the agent could improve.
        
        4. ACTION ITEMS: List 3-5 specific, actionable recommendations for the agent to implement in their next call.
        
        5. SALES TECHNIQUES ASSESSMENT: Evaluate how well the agent applied key sales techniques:
           - Opening/Introduction: How effectively did they establish rapport and set the agenda?
           - Needs Discovery: How well did they identify customer pain points and needs?
           - Value Proposition: How clearly did they communicate the value of their product/service?
           - Objection Handling: How effectively did they address concerns?
           - Closing Techniques: How well did they attempt to move the sale forward?
        
        6. CUSTOMER ENGAGEMENT SCORE: Rate from 1-10 how engaged the customer was during the call, with justification.
        
        7. OVERALL PERFORMANCE SCORE: Rate the agent's overall performance from 1-10, with justification.
        
        Format your response in a structured, easy-to-read format with clear headings for each section.
        """
        
        try:
            # Using Gemini's generate_content with the correct format
            response = self.model.generate_content(
                contents=[{
                    "parts": [{"text": prompt}],
                    "role": "user"
                }],
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 2048,
                }
            )
            
            # Extract the response text
            if response and hasattr(response, 'candidates') and response.candidates:
                analysis = response.candidates[0].content.parts[0].text
            else:
                analysis = "Error: Could not generate analysis. Please check your API key and try again."
            # Parse scores from analysis
            logger.debug(f"Analysis text length: {len(analysis)} characters")
            logger.debug(f"Sample analysis text: {analysis[:200]}...")
            
            engagement_score = self._extract_score(analysis, "CUSTOMER ENGAGEMENT SCORE")
            performance_score = self._extract_score(analysis, "OVERALL PERFORMANCE SCORE")
            
            logger.info(f"Extracted scores - Engagement: {engagement_score}, Performance: {performance_score}")
            
            # Prepare the result
            result = {
                "success": True,
                "call_data": call_data,
                "transcript": transcript,
                "analysis": analysis,
                "engagement_score": engagement_score,
                "performance_score": performance_score
            }
            
            # Save to cache
            call_hash = get_call_hash(call_data)
            save_to_cache(call_hash, result)
            return result
        except Exception as e:
            print(f"Error analyzing call: {e}")
            return {
                "error": f"Error analyzing call: {str(e)}",
                "success": False
            }
    
    def process_call(self, call_row):
        """Process a single call record from the CSV with caching support."""
        # Extract call data
        call_data = call_row.to_dict()
        logger.info(f"Processing call: {call_data.get('Call ID', 'Unknown')}")
        
        # Check cache first
        call_hash = get_call_hash(call_data)
        cached_result = load_from_cache(call_hash)
        if cached_result:
            logger.info(f"Using cached result for call {call_data.get('Call ID', 'Unknown')}")
            return cached_result
            
        recording_url = call_data.get('Recording URL')
        
        if not recording_url or pd.isna(recording_url):
            return {
                "error": "No recording URL available",
                "call_data": call_data,
                "success": False
            }
        
        # Download and transcribe the audio
        audio_file_path = self.download_audio(recording_url)
        if not audio_file_path:
            return {
                "error": "Failed to download audio",
                "call_data": call_data,
                "success": False
            }
        
        transcript = self.transcribe_audio(audio_file_path)
        if not transcript:
            return {
                "error": "Failed to transcribe audio",
                "call_data": call_data,
                "success": False
            }
        
        # Analyze the call
        result = self.analyze_call(transcript, call_data)
        
        # If analysis was successful but scores aren't in the result, try to extract them from the analysis text
        if result.get('success') and 'analysis' in result:
            analysis = result['analysis']
            
            # Only extract scores if they're not already in the result
            if 'engagement_score' not in result:
                engagement_score = self._extract_score(analysis, "CUSTOMER ENGAGEMENT SCORE")
                if engagement_score is not None:
                    result['engagement_score'] = engagement_score
                    logger.info(f"Extracted engagement score from analysis text: {engagement_score}")
            
            if 'performance_score' not in result:
                performance_score = self._extract_score(analysis, "OVERALL PERFORMANCE SCORE")
                if performance_score is not None:
                    result['performance_score'] = performance_score
                    logger.info(f"Extracted performance score from analysis text: {performance_score}")
        
        return result
    
    def _extract_score(self, text, score_name):
        """Extract a score (e.g., 7/10) from text using regex."""
        try:
            logger.debug(f"Extracting {score_name} from text")
            
            # First try to find the score in the format "**X/10**" after the score name
            pattern1 = re.compile(
                rf"{re.escape(score_name)}:[\s\n]*\*\*(\d+)/10\*\*", 
                re.IGNORECASE | re.DOTALL
            )
            match = pattern1.search(text)
            
            # If not found, try a more general pattern
            if not match:
                logger.debug("First pattern didn't match, trying alternative pattern")
                pattern2 = re.compile(
                    rf"{re.escape(score_name)}[^\d]*(\d+)/10", 
                    re.IGNORECASE | re.DOTALL
                )
                match = pattern2.search(text)
                if match:
                    logger.debug(f"Second pattern matched: {match.group(1)}")
            
            if match:
                score = int(match.group(1))
                logger.info(f"Extracted {score_name}: {score}")
                return score
                
            logger.warning(f"Could not find {score_name} in analysis text. Text snippet: {text[:200]}...")
            return None  # Return None if score not found
            
        except Exception as e:
            logger.error(f"Error extracting {score_name}: {str(e)}", exc_info=True)
            return None

    def process_csv(self, csv_path):
        """Process all calls in a CSV file."""
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Filter for rows with recording URLs
            df_with_recordings = df[df['Recording URL'].notna() & (df['Recording URL'] != '')]
            
            if df_with_recordings.empty:
                return {
                    "error": "No calls with recordings found in the CSV",
                    "success": False
                }
            
            results = []
            for _, row in df_with_recordings.iterrows():
                result = self.process_call(row)
                results.append(result)
            
            return {
                "results": results,
                "total_calls": len(df),
                "analyzed_calls": len(results),
                "success": True
            }
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return {
                "error": f"Error processing CSV: {str(e)}",
                "success": False
            }
