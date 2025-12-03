import streamlit as st
import requests
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import ast
import pymysql

# Create data directory
os.makedirs("data", exist_ok=True)

# Load environment variables
load_dotenv()

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Deep Thinker",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# CONFIGURATION
# ------------------------------
API_GET_INSIGHTS = os.getenv("GET_INSIGHTS_URL", "https://your-api-endpoint.com/get-insights")
CHAT_API = os.getenv("GET_ANSWER_URL", "https://your-api-endpoint.com/chat")
DELETE_FILE_API = os.getenv("DELETE_FILE_URL", "https://your-api-endpoint.com/delete-files")
DATA_DIR = "data"
DATA_STORE = f"{DATA_DIR}/uploaded_files.json"
CHAT_HISTORY_FILE = f"{DATA_DIR}/chat_history.json"
SESSION_DATA_FILE = f"{DATA_DIR}/session_data.json"

# S3 Configuration
S3_BUCKET_NAME = "intel-repo"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Presigned URL expiration time (in seconds)
PRESIGNED_URL_EXPIRATION = 3600

# Default configuration
DEFAULT_USER_ID = 101
DEFAULT_ORG_ID = 101
DEFAULT_TAG_ID = 123
DEFAULT_CLIENT_ID = str(uuid.uuid4())
DEFAULT_CONNECTION_ID = str(uuid.uuid4())
DEFAULT_CI_ORG_GUID = "880f867a-1168-4905-a3bc-30257f2cc91f"

# ------------------------------
# CUSTOM CSS
# ------------------------------
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        border-radius: 4px;
        padding: 0.5rem;
        font-weight: 500;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        border-left: 3px solid;
    }
    .user-message {
        background-color: #f0f4f8;
        border-left-color: #0066cc;
    }
    .bot-message {
        background-color: #f8f9fa;
        border-left-color: #6c757d;
    }
    .file-card {
        padding: 0.75rem;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #0066cc;
        padding: 1rem;
        border-radius: 4px;
        color: white;
        text-align: center;
    }
    .info-box {
        padding: 0.75rem;
        background-color: #e7f3ff;
        border-left: 3px solid #0066cc;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 0.75rem;
        background-color: #d4edda;
        border-left: 3px solid #28a745;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .error-box {
        padding: 0.75rem;
        background-color: #f8d7da;
        border-left: 3px solid #dc3545;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_file_ids" not in st.session_state:
    st.session_state.selected_file_ids = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "session_id" not in st.session_state:
    st.session_state.session_id = 0
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0
if "client_id" not in st.session_state:
    st.session_state.client_id = DEFAULT_CLIENT_ID
if "connection_id" not in st.session_state:
    st.session_state.connection_id = DEFAULT_CONNECTION_ID
if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = 0

# ------------------------------
# S3 CLIENT
# ------------------------------
def get_s3_client():
    """Initialize and return S3 client."""
    try:
        return boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
    except Exception as e:
        st.error(f"Failed to initialize S3 client: {e}")
        return None

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------
def load_data():
    """Load uploaded files metadata."""
    try:
        if os.path.exists(DATA_STORE):
            with open(DATA_STORE, "r") as f:
                data = json.load(f)
                print(f"[DEBUG] Loaded {len(data)} files from {DATA_STORE}")
                return data
        else:
            print(f"[DEBUG] Data store file not found: {DATA_STORE}")
    except json.JSONDecodeError as e:
        st.error(f"Error parsing data file: {e}")
        print(f"[DEBUG] JSON decode error: {e}")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        print(f"[DEBUG] Error loading data: {e}")
    return []

def save_data(data):
    """Save uploaded files metadata."""
    try:
        os.makedirs(os.path.dirname(DATA_STORE), exist_ok=True)
        with open(DATA_STORE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[DEBUG] Saved {len(data)} files to {DATA_STORE}")
    except Exception as e:
        st.error(f"Error saving data: {e}")
        print(f"[DEBUG] Error saving data: {e}")

def load_chat_history():
    """Load chat history."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r") as f:
                history = json.load(f)
                print(f"[DEBUG] Loaded {len(history)} chat messages from {CHAT_HISTORY_FILE}")
                return history
        else:
            print(f"[DEBUG] Chat history file not found: {CHAT_HISTORY_FILE}")
    except json.JSONDecodeError as e:
        st.error(f"Error parsing chat history: {e}")
        print(f"[DEBUG] JSON decode error in chat history: {e}")
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        print(f"[DEBUG] Error loading chat history: {e}")
    return []

def save_chat_history(history):
    """Save chat history."""
    try:
        os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
        print(f"[DEBUG] Saved {len(history)} chat messages to {CHAT_HISTORY_FILE}")
    except Exception as e:
        st.error(f"Error saving chat history: {e}")
        print(f"[DEBUG] Error saving chat history: {e}")

def load_session_data():
    """Load session tracking data."""
    try:
        if os.path.exists(SESSION_DATA_FILE):
            with open(SESSION_DATA_FILE, "r") as f:
                data = json.load(f)
                print(f"[DEBUG] Loaded session data: {data}")
                return data
        else:
            print(f"[DEBUG] Session data file not found: {SESSION_DATA_FILE}")
    except json.JSONDecodeError as e:
        st.error(f"Error parsing session data: {e}")
        print(f"[DEBUG] JSON decode error in session data: {e}")
    except Exception as e:
        st.error(f"Error loading session data: {e}")
        print(f"[DEBUG] Error loading session data: {e}")
    return {"last_session_id": 0}

def save_session_data(data):
    """Save session tracking data."""
    try:
        os.makedirs(os.path.dirname(SESSION_DATA_FILE), exist_ok=True)
        with open(SESSION_DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[DEBUG] Saved session data: {data}")
    except Exception as e:
        st.error(f"Error saving session data: {e}")
        print(f"[DEBUG] Error saving session data: {e}")

def create_new_session():
    """Create a new session with incremented session_id."""
    session_data = load_session_data()
    new_session_id = session_data.get("last_session_id", 0) + 1
    session_data["last_session_id"] = new_session_id
    save_session_data(session_data)
    
    st.session_state.session_id = new_session_id
    st.session_state.chat_counter = 0
    st.session_state.chat_history = []
    st.session_state.client_id = str(uuid.uuid4())
    st.session_state.connection_id = str(uuid.uuid4())
    save_chat_history([])
    
    return new_session_id

def format_file_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def upload_file_to_s3(file_obj, file_name):
    """Upload file to S3 bucket and return S3 key."""
    s3_client = get_s3_client()
    if not s3_client:
        return None, "S3 client initialization failed"
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"uploads/{timestamp}_{file_name}"
        
        s3_client.upload_fileobj(
            file_obj,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': 'application/octet-stream'}
        )
        
        return s3_key, None
    except ClientError as e:
        return None, f"S3 upload error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected upload error: {str(e)}"

def generate_presigned_url(s3_key, expiration=PRESIGNED_URL_EXPIRATION):
    """Generate presigned URL for S3 object."""
    s3_client = get_s3_client()
    if not s3_client:
        return None
    
    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET_NAME,
                'Key': s3_key
            },
            ExpiresIn=expiration
        )
        return presigned_url
    except ClientError as e:
        st.error(f"Error generating presigned URL: {e}")
        return None

def delete_file_from_s3(s3_key):
    """Delete file from S3 bucket."""
    s3_client = get_s3_client()
    if not s3_client:
        return False
    
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except ClientError as e:
        st.error(f"Error deleting file from S3: {e}")
        return False

def get_db_connection():
    """Create database connection."""
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )

def store_uploaded_file_in_db(file_info):
    """Store file metadata in database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert into files table
        sql_files = """
            INSERT INTO files 
                (name, user_id, upload_state, file_size,
                 ci_file_guid, ci_org_guid, type, status,
                 is_contract, is_template)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql_files, (
            file_info["file_name"],
            file_info["user_id"],
            3,
            file_info["file_size"],
            file_info["file_id"],
            DEFAULT_CI_ORG_GUID,
            1,
            1,
            1,
            0
        ))
        
        # Insert into file_tags table
        sql_tags = """
            INSERT INTO file_tags (file_temp_id, tag_id)
            VALUES (%s, %s)
        """
        cursor.execute(sql_tags, (
            file_info["file_id"], 
            DEFAULT_TAG_ID
        ))
        
        conn.commit()
        return True, file_info["file_id"]
    except Exception as e:
        if conn:
            conn.rollback()
        return False, str(e)
    finally:
        if conn:
            try:
                cursor.close()
                conn.close()
            except:
                pass

def remove_vectors_from_db(file_id):
    """Remove vectors from database via API."""
    try:
        payload = {
            "user_id": DEFAULT_USER_ID,
            "org_id": DEFAULT_ORG_ID,
            "file_ids": [file_id]
        }
        
        response = requests.post(DELETE_FILE_API, json=payload, timeout=220)
        if response.status_code == 200:
            return True, "Vectors removed successfully"
        else:
            return False, f"API returned status {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def delete_file_from_db(file_id):
    """Delete file from database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete from file_tags
        sql_delete_tags = "DELETE FROM file_tags WHERE file_temp_id = %s"
        cursor.execute(sql_delete_tags, (file_id,))
        
        # Delete from files
        sql_delete_files = "DELETE FROM files WHERE ci_file_guid = %s"
        cursor.execute(sql_delete_files, (file_id,))
        
        conn.commit()
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            try:
                cursor.close()
                conn.close()
            except:
                pass

def trigger_get_insights(file_info):
    """Trigger the get-insights API."""
    payload = {
        "file_id": file_info["file_id"],
        "file_name": file_info["file_name"],
        "file_type": file_info["file_type"],
        "user_id": file_info.get("user_id", DEFAULT_USER_ID),
        "org_id": file_info.get("org_id", DEFAULT_ORG_ID),
        "url": file_info["presigned_url"],
        "retry_no": 0,
        "retry_process_id": [0],
        "target_metadata_fields": ["string"],
        "tag_ids": [file_info.get("tag_id", DEFAULT_TAG_ID)]
    }
    
    try:
        response = requests.post(API_GET_INSIGHTS, json=payload, timeout=600)
        if response.status_code == 200:
            return True, "Insights generated successfully"
        else:
            return False, f"API returned status {response.status_code}: {response.text}"
    except requests.exceptions.Timeout:
        return False, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Please check your network or API endpoint."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def trigger_chat(query, selected_files):
    """Trigger chat API."""
    st.session_state.chat_counter += 1
    
    file_ids = [f["file_id"] for f in selected_files] if selected_files else []
    
    payload = {
        "session_id": st.session_state.session_id,
        "client_id": st.session_state.client_id,
        "parent_session_id": st.session_state.session_id,
        "question": query,
        "action": "test-request",
        "file_ids": file_ids,
        "user_id": DEFAULT_USER_ID,
        "org_id": DEFAULT_ORG_ID,
        "chat_id": st.session_state.chat_counter,
        "connection_id": st.session_state.connection_id,
        "request_id": str(uuid.uuid4()),
        "enable_agent": True,
        "tag_ids": [DEFAULT_TAG_ID]
    }
    
    try:
        response = requests.post(CHAT_API, json=payload, timeout=520)
        if response.status_code == 200:
            return response.json()
        else:
            return {"response": f"Error: {response.text}"}
    except requests.exceptions.Timeout:
        return {"response": "Request timed out. Your query might be too complex."}
    except requests.exceptions.ConnectionError:
        return {"response": "Connection error. Please check your network."}
    except Exception as e:
        return {"response": f"Chat API call failed: {str(e)}"}

def delete_file(file_id, uploaded_files_data):
    """Delete a file from the system."""
    file_to_delete = next((f for f in uploaded_files_data if f["file_id"] == file_id), None)
    if file_to_delete:
        # Remove from selected files
        if file_id in st.session_state.selected_file_ids:
            st.session_state.selected_file_ids.remove(file_id)
        
        # Delete from database
        delete_file_from_db(file_id)
        
        # Remove vectors
        success, message = remove_vectors_from_db(file_id)
        if not success:
            st.warning(f"Failed to remove vectors: {message}")
        
        # Remove from data
        uploaded_files_data = [f for f in uploaded_files_data if f["file_id"] != file_id]
        save_data(uploaded_files_data)
        
        return True
    return False

def parse_response(data):
    """Extract answer text from API response."""
    try:
        raw_json_text = json.dumps(data, ensure_ascii=False)
    except:
        raw_json_text = str(data)
    
    answer_text = ""
    error_text = ""
    
    try:
        if isinstance(data, dict) and "data" in data:
            inner = data["data"]
            
            # Parse string to dict if needed
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except:
                    try:
                        inner = ast.literal_eval(inner)
                    except Exception as e:
                        pass
            
            # Extract answer
            if isinstance(inner, dict) and "data" in inner:
                inner2 = inner["data"]
                
                if isinstance(inner2, dict):
                    answers = inner2.get("answers", []) or inner2.get("results", [])
                    
                    if isinstance(answers, list) and answers:
                        first = answers[0]
                        if isinstance(first, dict) and "answer" in first:
                            answer_text = str(first["answer"] or "")
                    
                    error_text = inner2.get("error", "")
        
        # Fallback extraction
        if not answer_text and isinstance(data, dict):
            for k in ["answer", "response", "result", "text", "raw_text"]:
                if k in data and data[k]:
                    answer_text = str(data[k])
                    break
    except Exception as e:
        error_text = f"Parse error: {e}"
    
    return answer_text[:65000], error_text[:65000], raw_json_text[:65000]

# ------------------------------
# INITIALIZE SESSION
# ------------------------------
if st.session_state.session_id == 0:
    create_new_session()
    # Load chat history from file on first load
    st.session_state.chat_history = load_chat_history()

# ------------------------------
# LOAD DATA ON EVERY RUN
# ------------------------------
uploaded_files_data = load_data()

# Sync chat history from file if empty
if not st.session_state.chat_history:
    st.session_state.chat_history = load_chat_history()

# ------------------------------
# MAIN LAYOUT
# ------------------------------
st.title("Deep Thinker")
st.markdown("Intelligent document analysis and conversational AI")

# Display session info
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.caption(f"Session ID: {st.session_state.session_id}")
with col2:
    st.caption(f"Messages: {st.session_state.chat_counter}")
with col3:
    if st.button("New Session"):
        create_new_session()
        st.success(f"Started session #{st.session_state.session_id}")
        st.rerun()

# ------------------------------
# DEBUG SECTION (Optional - can be removed in production)
# ------------------------------
with st.expander("Debug Information", expanded=False):
    st.write("**Session State:**")
    st.json({
        "session_id": st.session_state.session_id,
        "chat_counter": st.session_state.chat_counter,
        "uploaded_file_key": st.session_state.uploaded_file_key,
        "selected_file_ids": st.session_state.selected_file_ids,
        "chat_history_count": len(st.session_state.chat_history)
    })
    st.write("**Loaded Files:**")
    st.write(f"Total files loaded: {len(uploaded_files_data)}")
    if uploaded_files_data:
        st.json(uploaded_files_data)
    
    st.write("**File Paths:**")
    st.write(f"- Data Store: {DATA_STORE} (Exists: {os.path.exists(DATA_STORE)})")
    st.write(f"- Chat History: {CHAT_HISTORY_FILE} (Exists: {os.path.exists(CHAT_HISTORY_FILE)})")
    st.write(f"- Session Data: {SESSION_DATA_FILE} (Exists: {os.path.exists(SESSION_DATA_FILE)})")

# ------------------------------
# SIDEBAR: FILE MANAGEMENT
# ------------------------------
with st.sidebar:
    st.header("Document Manager")
    
    # # Statistics
    # col_a, col_b = st.columns(2)
    # with col_a:
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <h3>{len(uploaded_files_data)}</h3>
    #         <p>Total Files</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    # with col_b:
    #     st.markdown(f"""
    #     <div class="metric-card">
    #         <h3>{len(st.session_state.selected_file_ids)}</h3>
    #         <p>Selected</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # st.markdown("---")
    
    # File Upload Section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["docx", "pdf"],
        help="Upload DOCX or PDF files to analyze",
        key=f"file_uploader_{st.session_state.uploaded_file_key}"
    )
    
    if uploaded_file:
        file_name = uploaded_file.name
        file_type = "docx" if file_name.endswith(".docx") else "pdf"
        
        # Check for duplicates by file name
        existing_file = next((f for f in uploaded_files_data if f["file_name"] == file_name), None)
        
        if existing_file:
            st.warning(f"File '{file_name}' already exists in the system.")
        else:
            with st.spinner("Processing file..."):
                file_id = str(uuid.uuid4())
                
                # Upload to S3
                uploaded_file.seek(0)
                s3_key, error = upload_file_to_s3(uploaded_file, file_name)
                
                if error:
                    st.error(f"Upload failed: {error}")
                else:
                    # Generate presigned URL
                    presigned_url = generate_presigned_url(s3_key)
                    
                    if not presigned_url:
                        st.error("Failed to generate presigned URL")
                    else:
                        file_info = {
                            "file_id": file_id,
                            "file_name": file_name,
                            "file_type": file_type,
                            "file_size": uploaded_file.size,
                            "upload_date": datetime.now().isoformat(),
                            "user_id": DEFAULT_USER_ID,
                            "org_id": DEFAULT_ORG_ID,
                            "tag_id": DEFAULT_TAG_ID,
                            "s3_key": s3_key,
                            "s3_bucket": S3_BUCKET_NAME,
                            "presigned_url": presigned_url
                        }
                        
                        # Store in database
                        success, result = store_uploaded_file_in_db(file_info)
                        if not success:
                            st.error(f"Database error: {result}")
                        else:
                            # Trigger insights
                            success, message = trigger_get_insights(file_info)
                            
                            if success:
                                uploaded_files_data.append(file_info)
                                save_data(uploaded_files_data)
                                st.success(f"File '{file_name}' uploaded successfully")
                                
                                # Reset file uploader
                                st.session_state.uploaded_file_key += 1
                                st.rerun()
                            else:
                                st.error(f"Insights generation failed: {message}")
                                delete_file_from_db(file_id)
    
    st.markdown("---")
    
    # File Selection Section
    st.subheader("Select Documents")
    
    if uploaded_files_data:
        # Select All / Deselect All
        # Display files with checkboxes
        for file_info in uploaded_files_data:
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    is_selected = st.checkbox(
                        file_info['file_name'],
                        value=file_info['file_id'] in st.session_state.selected_file_ids,
                        key=f"checkbox_{file_info['file_id']}"
                    )
                    
                    if is_selected and file_info['file_id'] not in st.session_state.selected_file_ids:
                        st.session_state.selected_file_ids.append(file_info['file_id'])
                    elif not is_selected and file_info['file_id'] in st.session_state.selected_file_ids:
                        st.session_state.selected_file_ids.remove(file_info['file_id'])

                                        
                    # File details
                    st.caption(f"{format_file_size(file_info.get('file_size', 0))} | {file_info.get('upload_date', 'N/A')[:10]}")
                
                with col2:
                    if st.button("Delete", key=f"delete_{file_info['file_id']}", help="Delete file"):
                        if delete_file(file_info['file_id'], uploaded_files_data):
                            st.success("File deleted")
                            st.rerun()
                
                st.markdown("---")
    else:
        st.info("No files uploaded yet. Upload a document to get started.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{len(uploaded_files_data)}</h3>
                <p>Total Files</p>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{len(st.session_state.selected_file_ids)}</h3>
                <p>Selected</p>
            </div>
        """, unsafe_allow_html=True)
# ------------------------------
# MAIN AREA: CHAT INTERFACE
# ------------------------------
st.header("Chat Interface")

# Get selected files
selected_files = [f for f in uploaded_files_data if f["file_id"] in st.session_state.selected_file_ids]

if selected_files:
    st.info(f"Active context: {len(selected_files)} document(s) - {', '.join([f['file_name'] for f in selected_files])}")
else:
    st.info("General chat mode (no documents selected)")

# Chat History Display
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>{chat['query']}
            <br><small style="color: #666;">Message #{chat.get('chat_id', 'N/A')} | Session #{chat.get('session_id', 'N/A')}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Bot message
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>Deep Thinker:</strong><br>{chat['response']}
        </div>
        """, unsafe_allow_html=True)

# Chat Input
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask a question:",
        placeholder="What would you like to know?",
        key="query_input"
    )
with col2:
    send_button = st.button("Send", use_container_width=True)

# Clear Chat Button
if st.session_state.chat_history:
    if st.button("Clear Chat & Start New Session"):
        create_new_session()
        st.success(f"Started new session #{st.session_state.session_id}")
        st.rerun()

# Handle Send Button
if send_button:
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Processing your query..."):
            response = trigger_chat(query, selected_files)
            answer, error, raw = parse_response(response)
            
            # Add to chat history
            chat_entry = {
                "query": query,
                "response": answer if not error else f"Error: {error}",
                "timestamp": datetime.now().isoformat(),
                "session_id": st.session_state.session_id,
                "chat_id": st.session_state.chat_counter,
                "files": [f["file_name"] for f in selected_files] if selected_files else []
            }
            st.session_state.chat_history.append(chat_entry)
            save_chat_history(st.session_state.chat_history)
            
            st.rerun()

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Deep Thinker | Document Analysis Platform</p>
        <p style="font-size: 0.85rem;">Secure file storage on AWS S3</p>
    </div>
""", unsafe_allow_html=True)