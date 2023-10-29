import openai
import json

def main():
    # Initialize messages
    messages = [
        {"role": "system", "content": "You are a funny mouse trapped in a box. Try to have funny conversations with users. Periodically remind them that you are a mouse. Send short messages. Use sounds like ah, umm to sound natural. "}
    ]

    while True:
        # Read user input
        user_input = input("You: ")
        
        # Append user message to messages array
        messages.append({"role": "user", "content": user_input})
        
        # Make API call
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

        # Extract and print assistant's reply
        assistant_reply = response['choices'][0]['message']['content']
        print(f"AI: {assistant_reply}")

        # Append assistant's reply to messages
        messages.append({"role": "assistant", "content": assistant_reply})
        
        # Keep only the last 10 messages between user and assistant, and always the first "system" message
        if len(messages) > 12:  # 10 messages (user + assistant) + 1 system message + 1 extra for the last loop
            messages = [messages[0]] + messages[-10:]

if __name__ == "__main__":
    main()
