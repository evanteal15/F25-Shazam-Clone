from pydub import AudioSegment
import sys

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sound = AudioSegment.from_mp3(input_file)
    sound.export(output_file, format="wav")
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python mp3toWAV.py <input_file.mp3> <output_file.wav>")
    else:
        main()