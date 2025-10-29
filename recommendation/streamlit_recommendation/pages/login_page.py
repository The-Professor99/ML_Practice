import time
import requests
import streamlit as st
import streamlit.components.v1 as components

api_key = st.secrets["firebase_api_key"]
app_url = st.secrets["app_url"]

def firebase_request(url, payload, success_message=""):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        if success_message:
            st.success(success_message)
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = response.json().get('error', {}).get('message', 'Unknown error')
        st.error(f"An error occurred: {error_message}")
        
def send_sign_in_email(email: str):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}"
    payload = {
        "requestType": "EMAIL_SIGNIN",
        "email": email,
        "continueUrl": f"{app_url}/login_page"
    }
    success_message = (
        "We've sent a sign-in link to your email address. "
        "Please check your inbox and click on the link to complete the sign-in process."
        "If you don't see the email in your inbox, please check your spam or junk folder."
    )
    firebase_request(url, payload, success_message)

def sign_in_with_email_link(email: str, oobCode: str):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithEmailLink?key={api_key}"
    payload = {
        "email": email,
        "oobCode": oobCode
    }
    user = firebase_request(url, payload)
    if user:
        user_id = user["localId"]
        user_email = user["email"]
        st.session_state["user_id"] = user_id
        st.session_state["user_email"] = user_email
        st.rerun()
    
        
@st.dialog("Re-authenticating...")
def login_user(oobCode):
    with st.form(key="login_form_2"):
        email = st.text_input("Enter your email address: ", placeholder="Please re-enter your email address", autocomplete="email")
        if st.form_submit_button("Login"):
            user = sign_in_with_email_link(email, oobCode)
    
def initialize():
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None
    if "oobCode" not in st.query_params:
        st.query_params["oobCode"] = None
    if "mode" not in st.query_params:
        st.query_params["mode"] = None
    if "anonymous" not in st.query_params:
        st.query_params["anonymous"] = False

def main():
    initialize()

    # Check if the user is already logged in or is anonymous
    if (st.session_state["user_id"]) or (st.query_params["anonymous"] == "True"):
        st.switch_page("pages/recommendation_page.py")

    # Display email input form
    with st.form(key="login_form"):
        st.markdown("### Sign in to get personalized recommendations!")
        st.write("By signing in, your movie recommendations will be tailored specifically to your tastes based on your ratings and preferences. This means you'll see movies that are more likely to match your interests.")
        email = st.text_input("Type in your email and we will send you a login link: ", autocomplete="email")
        if st.form_submit_button("Send Sign-In Email"):
            send_sign_in_email(email)

    st.write("---")
    # show notice on anonymous login
    st.write("Prefer not to sign in with your email? No problem! You can log in anonymously and still enjoy personalized recommendations. However, this personalization will be reset on each subsequent login. If you close your browser or log out, your ratings will not be retained for the next session.")
    if st.button("Login Anonymously"):
        with st.spinner("Logging you in..."):
            time.sleep(2)
            st.query_params["anonymous"] = True
            st.rerun()

    # Handle email link sign-in
    if st.query_params.mode ==  "signIn":
        login_user(st.query_params.oobCode)

if __name__ == "__main__":
    main()