import streamlit as st
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import io
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Load the spaCy model for resume screening
nlp = spacy.load("en_core_web_sm")

# Connect to SQLite database
conn = sqlite3.connect('ats_system.db', check_same_thread=False)
cursor = conn.cursor()

# CSS Styling
st.markdown(
    """
    <style>
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1E1E1E;
            color: white;
        }
        .card {
        background-color: #f9f9f9;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
        }
        .card-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        /* Main title styling */
        .css-1v3fvcr {
            text-align: center;
            color: #4CAF50;
            font-family: 'Courier New', monospace;
        }
        /* Button styling */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            transition: 0.3s;
            font-weight: bold;
        }
        
        .stButton button:hover {
            background-color: #45a049;
        }
        /* Input fields styling */
        input, textarea, select {
            border: 1px solid #4CAF50;
            border-radius: 8px;
            padding: 8px;
            font-family: Arial, sans-serif;
            width: 100%;
            margin-bottom: 10px;
        }
        /* Sidebar selection styling */
        .css-18ni7ap {
            color: #1E1E1E;
            background-color: #4CAF50;
            border-radius: 8px;
            padding: 8px;
            font-weight: bold;
        }
        /* Table styling */
        .css-15zrgzn td, .css-15zrgzn th {
            text-align: left;
            padding: 8px;
            color: #333333;
            font-family: Arial, sans-serif;
        }
        /* Cards for categorized applications */
        .app-card {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f1f1f1;
            transition: box-shadow 0.3s ease;
        }
        .app-card:hover {
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
        }
        /* Success message styling */
        .stSuccess {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
        }
        /* Warning message styling */
        .stWarning {
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            color: #856404;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
        }
        /* Error message styling */
        .stError {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
        }
        /* Form inputs on focus */
        .form-input:focus {
            border-color: #45a049;
        }
        .dataframe {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        }
        .dataframe th {
            background-color: #f0f0f0;
            text-align: left;
        }
        .dataframe td {
            padding: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Database table setup
def setup_database():
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin', 'applicant'))
        )
    ''')
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            posted_on DATE NOT NULL,
            deadline DATE NOT NULL
        )
    ''')
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS applications (
            applicant_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            email TEXT,
            gender TEXT DEFAULT 'Unknown',
            job_id INTEGER,
            status TEXT DEFAULT 'Under Review',
            feedback TEXT DEFAULT '',
            submitted_on DATE NOT NULL,
            resume TEXT,
            match_score REAL DEFAULT 0.0,
            category TEXT DEFAULT 'Uncategorized',
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (job_id) REFERENCES jobs (job_id)
        )
    ''')
    conn.commit()
    # Ensure 'pdf_report' column exists
    try:
        cursor.execute("ALTER TABLE applications ADD COLUMN pdf_report TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN otp TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
# Function to calculate similarity score
def calculate_similarity(job_description, resume_text):
    vectorizer = CountVectorizer().fit_transform([job_description, resume_text])
    similarity_matrix = cosine_similarity(vectorizer)
    return similarity_matrix[0][1] * 100

def extract_skills_from_text(text):
    """Extracts only relevant skills from the provided text (resume or job description)."""
    doc = nlp(text.lower())
    extracted_skills = {token.text for token in doc if token.pos_ in {"NOUN", "PROPN"}}
    return extracted_skills

def generate_pdf(name, email, title, match_score, missing_skills, feedback):
    if not os.path.exists("feedback_reports"):
        os.makedirs("feedback_reports")
    file_path = os.path.join("feedback_reports", f"{name}_Feedback.pdf")
    c = canvas.Canvas(file_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(200, 750, "Application Feedback Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Applicant Name: {name}")
    c.drawString(100, 700, f"Email: {email}")
    c.drawString(100, 680, f"Job Title: {title}")
    c.drawString(100, 660, f"Match Score: {match_score:.2f}%")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 630, "Missing Skills:")
    c.setFont("Helvetica", 11)
    y_position = 610
    for skill in sorted(missing_skills):
        if y_position < 100:
            c.showPage()
            c.setFont("Helvetica", 11)
            y_position = 750
        c.drawString(120, y_position, f"- {skill}")
        y_position -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y_position - 20, "Feedback:")
    c.setFont("Helvetica", 11)
    c.drawString(120, y_position - 40, feedback)
    c.save()
    return file_path

def process_application(applicant_id):
    """Process a single applicant's job application."""
    cursor.execute("SELECT name, email, job_id, resume FROM applications WHERE applicant_id = ?", (applicant_id,))
    applicant_data = cursor.fetchone()
    if not applicant_data:
        print(f"No application found for applicant ID {applicant_id}.")
        return
    name, email, job_id, resume_text = applicant_data
    # Fetch the specific job description for the job applied for
    cursor.execute("SELECT title, description FROM jobs WHERE job_id = ?", (job_id,))
    job_data = cursor.fetchone()
    if not job_data:
        print(f"No job found for job ID {job_id}.")
        return
    job_title, job_description = job_data
    # Extract skills from job description and resume
    job_skills = extract_skills_from_text(job_description)
    resume_skills = extract_skills_from_text(resume_text)
    # Calculate match score and determine missing skills
    missing_skills = job_skills - resume_skills
    match_score = (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0
    # Generate feedback based on match score
    if match_score >= 50:
        status = 'Success'
        feedback = f"Congratulations! Your application matched with a score of {match_score:.2f}%."
    else:
        status = 'Rejected'
        feedback = f"Unfortunately, \n\n your application did not mieet our requirements due to missing key skills: {', '.join(missing_skills)}."
    # Generate and send PDF feedback
    pdf_path = generate_pdf(name, email, job_title, match_score, missing_skills, feedback, chart_path)
    send_email_with_pdf(name, email, pdf_path)
    # Update application status in the database
    cursor.execute("UPDATE applications SET status=?, feedback=?, match_score=? WHERE applicant_id=?", (status, feedback, match_score, applicant_id))
    conn.commit()
    
def send_email_with_pdf(name, email, pdf_path):
    sender_email = "nguaemmanuel190@gmail.com"
    sender_password = "vyok ebpu lctf xupr"
    subject = "Your Application Feedback Report"
    body = f"Hello {name},\n\nAttached is your application feedback report.\n\nBest regards,\nHT.co.ke"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    with open(pdf_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(pdf_path)}")
        msg.attach(part)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
        print(f"Email sent successfully to {email}")
    except Exception as e:
        print(f"Error sending email to {email}: {e}")

def screen_applications():
    applications = pd.read_sql("SELECT name, email, job_id, resume FROM applications WHERE status='Under Review'", conn)
    for _, row in applications.iterrows():
        name = row["name"]
        email = row["email"]
        job_id = row["job_id"]
        resume_text = row["resume"]
        #Fetch the job description from the database
        cursor.execute("SELECT description FROM jobs WHERE job_id = ?", (job_id,))
        job_data = cursor.fetchone()

        if not job_data:
            print(f"No job found for job ID {job_id}. Skipping application.")
            continue  

        job_description = job_data[0]  # Extract job description text
        #Now extract skills correctly
        job_skills = extract_skills_from_text(job_description)
        resume_skills = extract_skills_from_text(resume_text)

        if job_skills:
            match_score = (len(job_skills & resume_skills) / len(job_skills)) * 100
        else:
            match_score = 0
        
        missing_skills = job_skills - resume_skills
        
        if match_score >= 50:
            status = 'Success'
            feedback = f"Congratulations! Your application matched with a score of {match_score:.2f}%."
        else:
            status = 'Rejected'
            feedback = f"Unfortunately, \n your application did not meet our requirements due to missing key skills: {', '.join(missing_skills)}."
        
        pdf_path = generate_pdf(name, email, job_id, match_score, missing_skills, feedback)
        send_email_with_pdf(name, email, pdf_path)
        
        cursor.execute("UPDATE applications SET status=?, feedback=?, match_score=? WHERE name=?", 
                       (status, feedback, match_score, name))
        conn.commit()
# Register function for both roles
def register_user(role):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                       (username, hashed_password, role))
        conn.commit()
        st.success(f"Registration successful for {role}. Please log in.")
        st.balloons()
        time.sleep(2)
        
# Login function with redirection to the appropriate dashboard
def login(role):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        cursor.execute("SELECT * FROM users WHERE username = ? AND role = ?", (username, role))
        user = cursor.fetchone()
        if user and check_password_hash(user[2], password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            st.session_state["user_id"] = user[0]
            st.success(f"Login successful for {role}. Redirecting...")
            st.balloons()
            time.sleep(2)
            st.rerun()
        else:
            st.error("Incorrect username or password.")
        
    if st.button("Forgot Password?"):
        reset_password()

def send_otp(email):
    """Generate and send OTP to user email for password reset."""
    otp = str(random.randint(100000, 999999))
    cursor.execute("UPDATE users SET otp=? WHERE email=?", (otp, email))
    conn.commit()
    
    sender_email = "nguaemmanuel190@gmail.com"
    sender_password = "vyok ebpu lctf xupr"
    subject = "Password Reset OTP"
    body = f"Your OTP for password reset is: {otp}. It is valid for 10 minutes."
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
        st.success("OTP sent successfully to your email.")
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")

def reset_password():
    """Reset user password using OTP verification."""
    st.subheader("Forgot Password?")
    email = st.text_input("Enter your registered email")
    
    if st.button("Send OTP"):
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        if user:
            send_otp(email)
            st.session_state["reset_email"] = email
        else:
            st.error("Email not found.")
    
    if "reset_email" in st.session_state:
        otp = st.text_input("Enter OTP")
        new_password = st.text_input("Enter New Password", type="password")
        
        if st.button("Reset Password"):
            cursor.execute("SELECT otp FROM users WHERE email=?", (st.session_state["reset_email"],))
            stored_otp = cursor.fetchone()
            
            if stored_otp and stored_otp[0] == otp:
                hashed_password = generate_password_hash(new_password)
                cursor.execute("UPDATE users SET password=?, otp=NULL WHERE email=?", (hashed_password, st.session_state["reset_email"]))
                conn.commit()
                st.success("Password reset successfully! You can now login with your new password.")
                del st.session_state["reset_email"]
            else:
                st.error("Invalid OTP. Please try again.")
    
def send_bulk_feedbackk(applicants, status):
    sender_email = "nguaemmanuel190@gmail.com"
    sender_password = "vyok ebpu lctf xupr"  # Use App Password if needed
    subject = f"Application Update - {status}"

    success_count = 0  # Track successful emails
    
    if applicants.empty:
        st.warning("⚠️ No applicants to send emails to.")
        return 0  # Ensure it returns an integer

    st.write("📧 Sending emails to", len(applicants), "applicants...")
    
    for _, row in applicants.iterrows():
        recipient_email = row['email']
        name = row['name']
        message = f"Dear {name}, Your application has been categorized as '{status}'."

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, msg.as_string())
                success_count += 1  # Increment count on successful email
            st.success(f"✅ Email successfully sent to {recipient_email}")
        except smtplib.SMTPAuthenticationError:
            st.error("🔒 SMTP Authentication Error: Please check your email credentials or use an App Password.")
            return 0
        except smtplib.SMTPRecipientsRefused:
            st.error(f"🚫 Invalid recipient address: {recipient_email}. Please verify the email format.")
        except smtplib.SMTPConnectError:
            st.error("🌐 Connection Error: Unable to reach the mail server. Check your internet connection.")
            return 0
        except smtplib.SMTPException as e:
            st.error(f"⚠️ SMTP Error: {e}")
            return 0
        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")
            return 0
    return success_count  # Always return an integer

# Admin dashboard: job management, application screening, feedback
def admin_dashboard():
    st.sidebar.title("Admin Dashboard")
    st.sidebar.title(f"Welcome, {st.session_state['username']} !!!")
    action = st.sidebar.radio("Options", ["Dashboard","Post Job", "Manage Jobs", "View Applications","Categorized Applications","Generate Reports"])

    if action == "Dashboard":
        st.title("Data Visualizations")

        # Set up columns for side-by-side visualization
        col1, col2 = st.columns(2)

        # Visualization 1: Total Applications by Status (Pie Chart)
        with col1:
            st.markdown("<div class='card'><div class='card-title'>Total Applications by Status</div>", unsafe_allow_html=True)
            app_status = pd.read_sql("SELECT status, COUNT(*) as count FROM applications GROUP BY status", conn)
            fig1, ax1 = plt.subplots()
            ax1.pie(app_status['count'], labels=app_status['status'], autopct='%1.0f%%', startangle=140)  # Removed decimals
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        # Visualization 2: Gender Distribution of Applicants (Bar Chart)
        with col2:
            st.markdown("<div class='card'><div class='card-title'>Gender Distribution of Applicants</div>", unsafe_allow_html=True)
            gender_data = pd.read_sql("SELECT gender, COUNT(*) as count FROM applications GROUP BY gender", conn)
            fig2, ax2 = plt.subplots()
            sns.barplot(x='gender', y='count', data=gender_data, ax=ax2)
            ax2.set_xlabel("Gender")
            ax2.set_ylabel("Count")

            # Set y-axis to show integer values only
            ax2.yaxis.get_major_locator().set_params(integer=True)
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

    elif action == "Post Job":
        st.subheader("Post a New Job")

        # Initialize session state variables before creating widgets
        if "job_title" not in st.session_state:
            st.session_state["job_title"] = ""
        if "job_description" not in st.session_state:
            st.session_state["job_description"] = ""
        if "job_deadline" not in st.session_state:
            st.session_state["job_deadline"] = None

        # Input fields with session state
        title = st.text_input("Job Title", value=st.session_state["job_title"], key="job_title").strip()
        description = st.text_area("Job Description", value=st.session_state["job_description"], key="job_description").strip()
        deadline = st.date_input("Application Deadline", key="job_deadline")

        # Post Job button with validation
        if st.button("Post Job"):
            errors = []  # Track missing fields

            # Validate required fields
            if not title:
                errors.append("Job Title")
            if not description:
                errors.append("Job Description")
            if not deadline:
                errors.append("Application Deadline")

            if errors:
                st.error(f"⚠️ Please complete the following fields: **{', '.join(errors)}**")
                st.stop()

            # Check if the job title already exists
            existing_job = cursor.execute("SELECT COUNT(*) FROM jobs WHERE title = ?", (title,)).fetchone()[0]

            if existing_job > 0:
                st.warning("⚠️ This job title already exists. Please post a different job or update the existing one.")
                st.stop()

            # If all fields are valid and job does not exist, insert into database
            posted_on = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                "INSERT INTO jobs (title, description, posted_on, deadline) VALUES (?, ?, ?, ?)",
                (title, description, posted_on, deadline),
            )
            conn.commit()

            st.success("🎉 Job posted successfully!")

            # Use `st.session_state.clear()` to reset form fields
            for key in ["job_title", "job_description", "job_deadline"]:
                del st.session_state[key]  # Properly removes the session variable

            st.rerun()  # Refresh the page

    elif action == "Manage Jobs":
        # Fetch the total number of jobs available
        jobs_df = pd.read_sql("SELECT job_id, title, description, posted_on, deadline FROM jobs", conn)
        total_jobs = len(jobs_df)
        st.subheader(f"Current Job Openings ({total_jobs})")

        # Text input for search query
        search_query = st.text_input("🔍 Search Job by Title", "", key="search_job_title")

        # Dynamically filter jobs if search_query is not empty
        if search_query:
            jobs_df = jobs_df[jobs_df["title"].str.contains(search_query, case=False, na=False)]
        
        if jobs_df.empty:
            st.warning("No job postings available.")
        else:
            for index, row in jobs_df.iterrows():
                with st.expander(f"📌 {row['title']} (Deadline: {row['deadline']})"):
                    st.write(f"**Posted On:** {row['posted_on']}")
                    st.write(f"**Description:** {row['description']}")

                    col1, col2 = st.columns([1, 1])

                    # Edit Button
                    with col1:
                        if st.button("✏️ Edit", key=f"edit_{row['job_id']}"):
                            st.session_state["edit_job_id"] = row["job_id"]
                            st.session_state["edit_title"] = row["title"]
                            st.session_state["edit_description"] = row["description"]
                            st.session_state["edit_deadline"] = row["deadline"]
                            st.session_state["editing"] = True
                            st.rerun()

                    # Delete Button
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{row['job_id']}"):
                            cursor.execute("DELETE FROM jobs WHERE job_id=?", (row["job_id"],))
                            conn.commit()
                            st.success("Job deleted successfully!")
                            st.warning(f"Deleted job: {row['title']}")
                            time.sleep(2)
                            st.rerun()

        # If an edit button is clicked, show the editing form
        if st.session_state.get("editing", False):
            st.subheader("Edit Job Posting")

            new_title = st.text_input("Job Title", value=st.session_state["edit_title"])
            new_description = st.text_area("Job Description", value=st.session_state["edit_description"])
            new_deadline = st.date_input("Application Deadline", value=pd.to_datetime(st.session_state["edit_deadline"]))

            if st.button("Update Job"):
                cursor.execute("UPDATE jobs SET title=?, description=?, deadline=? WHERE job_id=?", 
                            (new_title, new_description, new_deadline, st.session_state["edit_job_id"]))
                conn.commit()
                st.success("Job updated successfully!")
                st.session_state["editing"] = False
                time.sleep(2)
                st.rerun()


    elif action == "View Applications":
        st.subheader("Applications Overview")
        screen_applications()
        st.subheader("Bias Detection Analysis")
    
        if st.button("Run Bias Detection"):
            detect_bias()
        with st.expander("Success"):
            success_apps = pd.read_sql("SELECT name, email, gender, job_id, status, feedback, resume, match_score FROM applications WHERE status='Success'", conn)
            st.write(success_apps)  
            
            if st.button("📩 Send to Stage 2"):
                if not success_apps.empty:
                    sent_emails = send_bulk_feedbackk(success_apps, "Success") or 0  
                    categorize_applications()
                    conn.commit()
                    if sent_emails > 0:  #Prevent NoneType error
                        st.success(f"🎉{sent_emails} Applications successfully sent to Stage 2 for further screening.")
                    else:
                        st.warning("⚠️Emails were not sent. Check SMTP settings.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.warning("⚠️No applications to send.")

        with st.expander("Rejected"):
            rejected_apps = pd.read_sql("SELECT name, email, gender, job_id, status, feedback, resume, match_score FROM applications WHERE status='Rejected'", conn)
            st.write(rejected_apps)
            if st.button("📩 Send Rejection Emails"):
                send_bulk_feedback(rejected_apps, "Rejected")
                for _, row in rejected_apps.iterrows():
                    name = row["name"]
                    email = row["email"]
                    job_id = row["job_id"]
                    match_score = row["match_score"]
                    feedback = row["feedback"]
                    
                    # Fetch job description from the database using job_id
                    cursor.execute("SELECT description FROM jobs WHERE job_id = ?", (job_id,))
                    job_data = cursor.fetchone()
                    job_description = job_data[0] if job_data else ""
                    
                    # Extract skills from resume and job description
                    resume_skills = extract_skills_from_text(row["resume"])
                    job_skills = extract_skills_from_text(job_description)  # Use job_description here
                    
                    missing_skills = job_skills - resume_skills

                    pdf_path = generate_pdf(name, email, job_id, match_score, missing_skills, feedback)
                    send_email_with_pdf(name, email, pdf_path)
                    
                    time.sleep(2)
                    st.rerun()
                else:
                    st.warning("⚠️No applications to send.")

        with st.expander("📩 Invite Candidates for Reassessment"):

                # Dynamic Filtering
                lower_threshold = st.slider("Lower Bound Match Score", 0, 100, 40)
                upper_threshold = st.slider("Upper Bound Match Score", 0, 100, 50)

                # Fetch candidates from the database based on the score range
                invite_candidates = pd.read_sql(
                    "SELECT name, email, gender, job_id, status, feedback, resume, match_score FROM applications WHERE status='Rejected' AND match_score BETWEEN ? AND ?",
                    conn,
                    params=(lower_threshold, upper_threshold),
                )

                if not invite_candidates.empty:
                    st.write(f"Showing candidates with match scores between {lower_threshold} and {upper_threshold}:")

                    # Add checkboxes for selection
                    selected_candidates = []
                    for index, row in invite_candidates.iterrows():
                        if st.checkbox(f"Select {row['name']}", key=f"select_{index}"):
                            selected_candidates.append(row)

                    # Display selected candidates
                    if selected_candidates:
                        st.write("**Selected Candidates:**")
                        selected_df = pd.DataFrame(selected_candidates)[["name", "email", "gender", "feedback", "match_score", "status"]]
                        st.dataframe(selected_df, width=None)

                        # Bulk Send Invitations
                        if st.button("📩 Send Invitations"):
                            for row in selected_candidates:
                                name, email, job_id, match_score, feedback, resume = row["name"], row["email"], row["job_id"], row["match_score"], row["feedback"], row["resume"]

                                # Fetch job description from the database
                                cursor.execute("SELECT description FROM jobs WHERE job_id = ?", (job_id,))
                                job_data = cursor.fetchone()
                                job_description = job_data[0] if job_data else ""

                                # Extract skills from resume and job description
                                resume_skills = extract_skills_from_text(resume)
                                job_skills = extract_skills_from_text(job_description)
                                missing_skills = job_skills - resume_skills

                                # Generate PDF with missing skills and feedback
                                pdf_path = generate_pdf(name, "Job Title", match_score, resume_skills, missing_skills, feedback)

                                # Send an invitation email with PDF
                                send_invitation_email(name, email, pdf_path)
                                time.sleep(2)  # Prevents overloading SMTP server

                            st.success("✅ Invitations sent successfully!")

                    else:
                        st.warning("⚠️ No candidates selected. Please select candidates to send invitations.")

                else:
                    st.info("📌 No candidates found for invitation within the selected score range.")

    elif action == "Generate Reports":
        st.subheader("Applicant Reports")
        report_df = pd.read_sql("SELECT name, email, gender, job_id, status, feedback, submitted_on, resume, match_score FROM applications", conn)
        st.write(report_df)
        st.download_button("Download Report as CSV", report_df.to_csv(index=False), "report.csv")

    if action == "Categorized Applications":
        view_categorized_applications()    



SMTP_SERVER = "smtp.gmail.com"  # Change if using another provider
SMTP_PORT = 587  # Standard TLS port
SENDER_EMAIL = "nguaemmanuel190@gmail.com"
SENDER_PASSWORD = "vyok ebpu lctf xupr"

def send_invitation_email(name, email, pdf_path):
    try:
        # Create email message
        msg = EmailMessage()
        msg["Subject"] = "Invitation for Interview - Job Opportunity"
        msg["From"] = SENDER_EMAIL
        msg["To"] = email

        # Email body (Updated for Interview Invitation)
        msg.set_content(
            f"""
            Dear {name},

            After reviewing your application, we are pleased to invite you for an **in-person interview** for the role.

            📅 **Interview Date:** {interview_date}  
            📍 **Location:** {interview_location}  

            Please find attached a detailed report highlighting our evaluation of your skills.  

            Kindly confirm your availability by replying to this email.

            Best regards,  
            Hiring Team
            """
        )

        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print(f"✅ Interview invitation sent to {email}")

    except Exception as e:
        print(f"❌ Error sending interview invitation to {email}: {e}")

def detect_bias():
    # Load application data
    df = pd.read_sql("SELECT gender, match_score, status FROM applications", conn)
    
    if df.empty:
        st.warning("No applications available for bias analysis.")
        return
    # Calculate the selection rate by gender
    metric_frame = MetricFrame(
        metrics=selection_rate,
        y_true=df['status'] == 'Success',
        y_pred=df['match_score'] >= 50,  # Assuming 'Success' if match_score is >= 50
        sensitive_features=df['gender']
    )
    # Calculate demographic parity difference
    parity_difference = demographic_parity_difference(
        y_true=df['status'] == 'Success',
        y_pred=df['match_score'] >= 50,
        sensitive_features=df['gender']
    )
    st.write("Selection Rate by Gender:", metric_frame.by_group)
    st.write("Demographic Parity Difference:", parity_difference )

    if abs(parity_difference) > 0.1:
        st.warning("Potential bias detected! Significant difference in selection rates. A value closer to 0 means fairness, while a higher value suggests potential bias.")
# Approve application
def approve_application(applicant_id, username, match_score):
    # Update status to "Approved" and send a message to the applicant
    message = f"Hi {username}, your resume passed 1st stage. It’s now in the 2nd stage. We'll notify you if you qualify."
    cursor.execute("UPDATE applications SET status='Approved', feedback=? WHERE applicant_id=?", (message, applicant_id))
    conn.commit()
    st.success("Applicant approved and notified.")
    # Categorize based on match score
    if match_score >= 80:
        category = "Highly Fit"
    elif match_score >= 70:
        category = "Moderate Fit"
    elif match_score >= 50:
        category = "Low Fit"
    else:
        category = "Rejected"
    # Save the category for display on the categorized page
    cursor.execute("UPDATE applications SET category=? WHERE applicant_id=?", (category, applicant_id))
    conn.commit()

# Categorize applications based on match score
def categorize_applications():
    applications = cursor.execute("SELECT applicant_id, match_score FROM applications WHERE status='Success'").fetchall()
    for app_id, match_score in applications:
        if match_score >= 80:
            category = "Highly Fit"
        elif match_score >= 70:
            category = "Moderate Fit"
        elif match_score >= 50:
            category = "Low Fit"
        else:
            category = "Rejected"
        
        cursor.execute("UPDATE applications SET category=? WHERE applicant_id=?", (category, app_id))
    conn.commit()

def send_invitation_email(name, email, pdf_path):
    """Sends an invitation email to a rejected applicant for reassessment."""
    sender_email = os.getenv("nguaemmanuel190@gmail.com")
    sender_password = os.getenv("vyok ebpu lctf xupr")

    subject = "Reassessment Invitation for Your Job Application"
    body = f"""
    Dear {name},

    We appreciate your interest in the position. While your application did not meet the initial passing criteria, 
    we believe you have potential! Your match score was {match_score:.2f}%.

    We encourage you to enhance your skills and reapply for the opportunity. 
    Your feedback report with missing skills is attached to this email.

    Best regards,
    HR Team
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach the PDF Report
    with open(pdf_path, "rb") as attachment:
        part = MIMEApplication(attachment.read(), Name=os.path.basename(pdf_path))
        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(pdf_path)}"'
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
        st.success(f"✅ Invitation sent to {name} ({email})")
    except Exception as e:
        st.error(f"❌ Failed to send invitation to {name}: {e}")

# Dummy email_sent function to avoid errors
def email_sent(email):
    # Placeholder logic: Assume emails are not delivered initially
    return False


# Display categorized applications
def view_categorized_applications():
    st.title("Categorized Applications")
    
    # Categories and colors
    categories = [
        {"name": "Highly Fit Candidates", "color": "#228B22", "db_category": "Highly Fit"},
        {"name": "Moderate Fit Candidates", "color": "#90EE90", "db_category": "Moderate Fit"},
        {"name": "Low Fit Candidates", "color": "#CD5C5C", "db_category": "Low Fit"},
    ]
    
    # Layout for the cards
    cols = st.columns(len(categories))
    
    for idx, category in enumerate(categories):
        with cols[idx]:
            st.markdown(f"""
                <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: {category['color']}; text-align: center;">
                    <h3>{category['name']}</h3>
                </div>
            """, unsafe_allow_html=True)

            if st.button(f"View {category['name']} Candidates", key=f"view_{category['db_category']}"):
                st.session_state["clicked_category"] = category["db_category"]

    clicked_category = st.session_state.get("clicked_category", None)

    if clicked_category:
        st.subheader(f"Candidates for {clicked_category}")
        
        # Fetch applicants
        applicants = cursor.execute(
            "SELECT name, email, match_score FROM applications WHERE category=?", 
            (clicked_category,)
        ).fetchall()
        
        st.session_state["applicants"] = applicants

        if applicants:
            # Convert to Pandas DataFrame for better alignment
            import pandas as pd
            applicant_df = pd.DataFrame(applicants, columns=["Name", "Email", "Match Score"])
            
            # Add Email Status Column
            applicant_df["Email Status"] = applicant_df["Email"].apply(lambda email: "Delivered" if email_sent(email) else "Not Delivered")
            
            st.dataframe(applicant_df, hide_index=True)

            # Send Emails Button
            if st.button(f"Send Emails to All {clicked_category} Candidates"):
                send_ca_feedback(st.session_state["applicants"], "Success")
        else:
            st.info(f"No candidates found for {clicked_category}.")


def send_ca_feedback(applicants, status):
    sender_email = "nguaemmanuel190@gmail.com"
    sender_password = "vyok ebpu lctf xupr"

    # Ensure applicants exist before sending emails
    if isinstance(applicants, pd.DataFrame):  # If applicants is a DataFrame
        if applicants.empty:
            st.warning("No applicants found to send emails.")
            return
        else:
            applicants_list = applicants.to_records(index=False)  # Convert DataFrame to list of tuples
    elif isinstance(applicants, list):  # If applicants is already a list of tuples
        if not applicants:
            st.warning("No applicants found to send emails.")
            return
        applicants_list = applicants  # Use as is
    else:
        st.error("Invalid applicants data format.")
        return

    # Iterate through the list of applicants and send emails
    for app in applicants_list:
        try:
            if isinstance(app, tuple) and len(app) >= 4:  # Ensure correct tuple length
                app_id, name, email, score = app[:4]  # Extract required fields
                subject = f"Application {status} Notification"
                body = f"Dear {name},\n\nYour application has been categorized as '{status}' with a match score of {score}%."
                message = f"Subject: {subject}\n\n{body}"

                send_email(email, name, message)  # Send email
        except Exception as e:
            st.error(f"Error processing applicant {name}: {e}")

# Function to validate email
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email)

# Function to send emails
def send_email(email, name, message):
    sender_email = "nguaemmanuel190@gmail.com"
    sender_password = "vyok ebpu lctf xupr"

    try:
        if not is_valid_email(email):
            st.error(f"Invalid email format: {email}")
            return

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message)

        st.success(f"Email sent successfully to {name} ({email})")
    except Exception as e:
        st.error(f"Failed to send email to {name}: {e}")

# Function to send bulk rejection emails
def send_bulk_feedback(applicants, status):
    if applicants.empty:
        st.warning("No applicants found to send emails.")
        return

    st.write("📧 Sending emails to", len(applicants), "applicants...")

    for _, row in applicants.iterrows():
        name, email, match_score = row["name"], row["email"], row["match_score"]

        subject = f"Application {status} Notification"
        body = f"Dear {name},\n\nYour application has been categorized as '{status}' with a match score of {match_score}%."
        message = f"Subject: {subject}\n\n{body}"

        send_email(email, name, message)

def status_badge(status):
    """Returns a slanted badge-like HTML element for application status."""
    colors = {"Under Review": "#FFC107", "Accepted": "#4CAF50", "Rejected": "#F44336"}
    return f"""<div style="position: absolute; right: -10px; top: 10px; 
                transform: rotate(15deg); background-color: {colors.get(status, '#607D8B')}; 
                color: white; padding: 5px 15px; border-radius: 5px; 
                font-size: 12px; font-weight: bold;">
                {status}</div>"""

def validate_email(email):
    """Checks if an email is valid."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email)

def applicant_dashboard():
    st.sidebar.title("Applicant Dashboard")
    st.sidebar.title(f"Welcome, {st.session_state['username']} !!!")
    action = st.sidebar.radio("Options", ["View Jobs", "Apply for a Job", "My Applications"])

    if action == "View Jobs":
        jobs = pd.read_sql("SELECT title, description, posted_on, deadline FROM jobs", conn)
        st.write(jobs)

    elif action == "Apply for a Job":
        st.warning("!NOTE: You cannot edit your application after submission once you apply you cannot apply again.")
        # Load job data
        job_data = pd.read_sql("SELECT job_id, title FROM jobs", conn)
        job_id = st.selectbox("Select Job", job_data["job_id"], format_func=lambda x: job_data[job_data["job_id"] == x]["title"].values[0])

        # Check if the user has already applied
        existing_application = cursor.execute(
            "SELECT COUNT(*) FROM applications WHERE user_id = ? AND job_id = ?", 
            (st.session_state["user_id"], job_id)
        ).fetchone()[0]

        if existing_application > 0:
            st.warning("⚠️ You have already applied for this job. You cannot apply again.")
            st.stop()

        # User Input Fields
        first_name = st.text_input("Full Name", key="first_name").strip()
        email = st.text_input("Email Address", key="email").strip()
        gender = st.selectbox("Gender", ["--Choose your gender--", "Male", "Female", "Other", "Prefer not to say"], key="gender")

        resume_upload = st.file_uploader("Upload Resume (PDF or Text)", type=["pdf", "txt"], key="resume_upload")
        resume_text = st.text_area("Or Paste your Resume here", key="resume_text").strip()  # Alternative if no file uploaded

        # Process Form Submission
        if st.button("Apply"):
            errors = []  # List to track missing fields

            # Validate fields and highlight missing ones
            if not first_name:
                errors.append("Full Name")
            if not email:
                errors.append("Email Address")
            if gender == "--Choose your gender--":
                errors.append("Gender")
            
            # Validate email format
            if email and not validate_email(email):
                st.error("❌ Invalid email format. Please enter a valid email address.")
                st.stop()

            # Ensure resume is provided
            if not resume_upload and not resume_text:
                errors.append("Resume (Upload or Paste)")

            # Show error messages highlighting missing fields
            if errors:
                st.error(f"❌ Please fill in the following fields: **{', '.join(errors)}**")
                st.stop()

            # Ensure only one resume option is used
            if resume_upload and resume_text:
                st.error("❌ Please provide either a resume file or paste your resume text, but not both.")
                st.stop()

            # Extract text from uploaded resume if provided
            if resume_upload:
                try:
                    resume_text = resume_upload.getvalue().decode("utf-8", errors="ignore")  # Safe decoding
                except Exception as e:
                    st.error(f"❌ Error processing uploaded file: {e}")
                    st.stop()

            # Save application to the database
            submitted_on = datetime.now().strftime("%Y-%m-%d")
            full_name = f"{first_name}"

            cursor.execute(
                "INSERT INTO applications (user_id, job_id, name, email, gender, submitted_on, resume) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                (st.session_state["user_id"], job_id, full_name, email, gender, submitted_on, resume_text)
            )
            conn.commit()
            st.success("✅ Application submitted successfully! You cannot edit your application after submission.")

            # Sending acknowledgment email
            sender_email = "nguaemmanuel190@gmail.com"
            sender_password = "vyok ebpu lctf xupr"
            subject = "Application Received"
            body = f"Dear {first_name},\n\nThank you for applying. Your application has been received, and you will be notified of every stage it undergoes."
            message = f"Subject: {subject}\n\n{body}"

            try:
                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.sendmail(sender_email, email, message)
                st.success("📩 A confirmation email has been sent to you.")
                time.sleep(2)
            except Exception as e:
                st.error(f"❌ Failed to send email. Error: {e}")

            st.rerun()

    elif action == "My Applications":
        applications = pd.read_sql("""
            SELECT a.job_id, j.title, j.deadline, a.submitted_on, a.status, a.feedback, a.email
            FROM applications a
            JOIN jobs j ON a.job_id = j.job_id
            WHERE a.user_id = ?
        """, conn, params=(st.session_state["user_id"],))

        if applications.empty:
            st.info("You have not applied for any jobs yet.")
        else:
            for index, row in applications.iterrows():
                with st.expander(f"📌 {row['title']}"):
                    st.markdown(f"""
                        <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 15px; margin-bottom: 10px; 
                                    background-color: #f9f9f9; position: relative; padding-top: 10px;">
                            {status_badge(row["status"])}
                            <h4 style="margin-bottom: 5px; background-color: #03A9F4; color: white; padding: 8px; 
                                        border-radius: 5px; display: inline-block;">
                                {row["title"]}
                            </h4>
                            <p style="margin: 5px 0;"><b>Applied on:</b> {row["submitted_on"]}</p>
                            <p style="margin: 5px 0;"><b>Deadline:</b> {row["deadline"]}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Display feedback based on status
                    if row["status"] == "Rejected":
                        st.error(f"❌ Your application was not successful.\n\n Feedback: {row['feedback']}")
                    elif row["status"] == "Success":
                        st.success(f"🎉 {row['feedback']} We will reach out to you via **{row['email']}**.")
                    elif row["status"] == "Under Review":
                        st.warning(f"⏳ Your application is under review. You will be notified at **{row['email']}**.")

# Main function to handle app flow with logout option
def main():
    setup_database()

    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        role = st.sidebar.selectbox("Role", ["admin", "applicant"])
        action = st.sidebar.selectbox("Action", ["Register", "Login"])
        if action == "Register":
            register_user(role)
        else:
            login(role)
    else:
        # Direct to respective dashboard based on role
        if st.session_state["role"] == "admin":
            admin_dashboard()
        elif st.session_state["role"] == "applicant":
            applicant_dashboard()
         # Add Logout button at the top of the dashboard
        if st.sidebar.button("Logout"):
            st.session_state.clear()  # Reset session state
            st.toast("🎉 **Goodbye!**")
            st.rerun()   # Refresh the app to reset to the login screen
if __name__ == "__main__":
    main()