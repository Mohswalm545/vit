import speech_recognition as sr
from pydub import AudioSegment

class SpeechRecognitionModel:
    def __init__(self, audio_path) -> None:
        self.audio_path = audio_path

    def transcribe_audio(self):
        # Initialize the recognizer
        r = sr.Recognizer()

        segment_duration = 3* 60 * 1000  # 3 minutes in milliseconds
        segments = []

        audio = AudioSegment.from_file(self.audio_path)
        total_duration = len(audio)
        num_segments = int(total_duration / segment_duration) + 1

        for i in range(num_segments):
            segment_start = i * segment_duration
            segment_end = min((i + 1) * segment_duration, total_duration)
            segment = audio[segment_start:segment_end]
            segments.append(segment)

        # Perform transcription for each segment
        transcriptions = []

        for segment in segments:
            with sr.AudioFile(segment.export(format="wav")) as source:
                segment_audio = r.record(source)
                text = r.recognize_google(segment_audio)
                transcriptions.append(text)

        # Combine the transcriptions into a single output
        output = " ".join(transcriptions)

        # Print the extracted text
        print(output)
        return output
