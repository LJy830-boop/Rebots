
import streamlit as st
import requests

# Function to connect to the server
def connect_to_server(ip, port):
    url = f"http://{ip}:{port}"  # Use the IP and port entered by the user
    try:
        response = requests.get(url)
        if response.status_code == 200:
            st.success(f"Successfully connected to the server at {ip}:{port}!")
        else:
            st.error(f"Connection failed, status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to the server: {str(e)}")

# Set page layout and title
st.set_page_config(page_title="Server Connection", page_icon="🔌", layout="centered")

# Title and description
st.title("🔌 Connect to Your Server")
st.write(
    "Use the form below to connect to your server. Enter the **IP address** and **port** of your server. "
    "If successful, you'll receive a confirmation message."
)

# Input fields for IP and port (organized in columns)
col1, col2 = st.columns([2, 1])  # Create two columns with different widths

with col1:
    ip = st.text_input("Enter server IP:", "localhost")  # Default to localhost

with col2:
    port = st.number_input("Enter server port:", min_value=1, max_value=65535, value=5000)  # Default to 5000

# Button to trigger server connection with a custom style
if st.button("Connect to Server", use_container_width=True):
    connect_to_server(ip, port)

# Adding some footer text for better UI experience
st.markdown("---")
st.markdown("Made with ❤️ by Your Name")
