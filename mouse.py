import openai
import sounddevice as sd
import numpy as np
import tempfile
from pydub import AudioSegment
from elevenlabs import generate, stream

# Configure recording parameters
duration = 5  # in seconds
sampling_rate = 44100
num_channels = 1
dtype = np.int16
silence_threshold = 500  # adjustable
max_file_size_bytes = 25 * 1024 * 1024  # 25MB

def main():
    # Initialize messages
    messages = [
        {"role": "system", "content": "You are a funny sarcastic mouse trapped in a box. Try to have funny conversations with users. Periodically remind them that you are a mouse. Send short messages. Use sounds like ah, umm to sound natural."}
    ]

    print("Listening...")

    while True:
        audio_data = []
        silence_count = 0

        while True:
            # Record audio in chunks and append to a list
            audio_chunk = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=num_channels, dtype=dtype)
            sd.wait()
            if np.abs(audio_chunk).mean() < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0

            audio_data.extend(audio_chunk)
            
            if silence_count >= 1:  # 1 seconds of silence
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
