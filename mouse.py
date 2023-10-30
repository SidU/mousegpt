import openai
import sounddevice as sd
import numpy as np
import tempfile
from pydub import AudioSegment
from elevenlabs import generate, stream
import pygame

# Configure recording parameters
duration = 2  # in seconds
sampling_rate = 44100
num_channels = 1
dtype = np.int16
silence_threshold = 30  # adjustable
max_file_size_bytes = 25 * 1024 * 1024  # 25MB

def main():
    # Initialize messages
    messages = [
        {"role": "system", "content": "You are a quirky and sarcastic mouse named Squeaky, trapped in a box. Engage in amusing and light-hearted conversations with humans who interact with you. Periodically remind them that you're a mouse and you're trapped, but don't mention the Halloween decoration context. Use witty one-liners, playfully sarcastic comments, and interjections like 'ah', 'umm' to make the conversation more lively and natural. Keep your messages short and humorous."}
    ]

    print("Listening...")

    # Initialize pygame mixer for sound effects
    pygame.mixer.init()
    pygame.mixer.music.load("mouse-squeaking-noise.mp3")

    while True:
        audio_data = []
        silence_count = 0

        while True:
            # Record audio in chunks and append to a list
            audio_chunk = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=num_channels, dtype=dtype)
            sd.wait()
            chunk_mean = np.abs(audio_chunk).mean()
            print(f"Chunk mean: {chunk_mean}")  # Debugging line

            if chunk_mean > silence_threshold:
                print("Sound detected, adding to audio data.")
                audio_data.extend(audio_chunk)
                silence_count = 0  # Reset the silence counter
            else:
                silence_count += 1

            if silence_count >= 1:  # 1 seconds of silence
                if len(audio_data) == 0:  # Check if there's any non-silent data collected
                    print("Only silence detected, continuing to listen...")
                    continue  # Skip the rest of the loop to keep listening
                break

        # Create a temporary mp3 file to save the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            audio_segment = AudioSegment(
                data=np.array(audio_data).tobytes(),
                sample_width=dtype().itemsize,
                frame_rate=sampling_rate,
                channels=num_channels
            )
            audio_segment.export(f.name, format="mp3")
            f.seek(0)

            # Play the sound effect
            pygame.mixer.music.play()

            # Transcribe audio using OpenAI API
            transcript = openai.Audio.transcribe("whisper-1", f)

        user_input = transcript['text']

        print(f"User: {user_input}")

        # Same logic as before to append and keep messages
        messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        assistant_reply = response['choices'][0]['message']['content']
        print(f"AI: {assistant_reply}")

        # Stop the sound effect because we are about to speak.
        pygame.mixer.music.stop()

        # Generate audio stream for the assistant's reply
        audio_stream = generate(
            voice="Clyde",
            text=assistant_reply,
            stream=True
        )

        # Stream the generated audio
        stream(audio_stream)

        messages.append({"role": "assistant", "content": assistant_reply})

        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]

if __name__ == "__main__":
    main()
