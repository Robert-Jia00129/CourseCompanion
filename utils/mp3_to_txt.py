import whisper

# Load model (tiny, base, small, medium, or large)
model = whisper.load_model("small")
def mp3_to_txt(mp3_paath):
    # Transcribe
    result = model.transcribe(file_path, language='en', verbose=True)
    print(result.keys())
    return result['text'], result['segments']

if __name__ == '__main__': 
    # file_path = '/Users/jiazhenghao/Downloads/a3578f0f-096d-54bc-8a8d-b1611b28c1b6.mp3'
    file_path = '/Users/jiazhenghao/Downloads/a3578f0f-096d-54bc-8a8d-b1611b28c1b6.m4a'
    
    text, segments = mp3_to_txt(file_path)
    with open('./transcript.txt', 'w') as f:
        f.write(text)
    print(text)
