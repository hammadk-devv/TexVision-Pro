import wave
import struct
import math

# Create a simple beep sound
sampleRate = 44100
durationSeconds = 0.3
frequencyHz = 800

with wave.open('static/alert.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sampleRate)
    
    for i in range(int(sampleRate * durationSeconds)):
        value = int(32767 * 0.3 * math.sin(2 * math.pi * frequencyHz * i / sampleRate))
        f.writeframes(struct.pack('h', value))

print("Alert sound created successfully!")
