# Install the necessary library
!pip install --upgrade google-generativeai

# Import the necessary library
import google.generativeai as genai
import os

# Set your API key from Google AI Studio
GEMINI_API_KEY = "AIzaSyCPNJcJxV_A2gWBB2z3DDL71TZTqmJ9jsw"

# Configure the Gemini API with your API key
if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not set. Please replace 'YOUR_API_KEY' with your actual key from Google AI Studio.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_response(user_input):
    """Generates a response based on user input."""
    try:
        # Use the model to generate content
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# Chatbot loop
def chatbot():
    print("Welcome to the Chatbot! Type 'exit' to stop the conversation.")

    while True:
        # Take user input
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("Chat ended.")
            break

        # Generate a response using the model
        bot_response = generate_response(user_input)

        # Display the bot's response
        print(f"Bot: {bot_response}\n")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
