import subprocess
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import librosa.display
from moviepy.editor import VideoFileClip, AudioFileClip
import plotly.graph_objects as go
import argparse


from hasher import preprocess_audio, compute_fft, convert_to_decibel, \
                   find_peaks, create_hashes, recognize_music

from DBcontrolSQLite import retrieve_song

def visualize_spectrogram_as_mp4(audio_path = "sample.wav"):
    hop_length = 512
    audio, sr = preprocess_audio(audio_path)
    frequencies, times, magnitudes = compute_fft(audio, sr, hop_length=hop_length)
    #print(magnitudes.shape)
    #print
    #exit(0)
    magnitudes = convert_to_decibel(magnitudes)

    constellation_map = find_peaks(frequencies, times, magnitudes)
    #print("--------- constellation")
    #print(constellation_map)
    hashes = create_hashes(constellation_map, sr=sr, explicit=True)

    #print("--------- hashes")
    #print(hashes)
    #exit(0)
    peak_times = [times[t] for t, f in constellation_map]
    peak_freqs = [f for t, f in constellation_map]


    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    img = librosa.display.specshow(
        magnitudes,
        sr=sr,
        #hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        cmap='inferno',
        ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set_title("Spectrogram with Playback Line")

    ax.scatter(peak_times, peak_freqs, color='white', marker='s', facecolor="none", s=20, label='Peaks')
    playback_line = ax.axvline(0, color='cyan', linewidth=2)

    hash_lines = []
    for h in hashes:
        #h["anchor"] = (h["anchor"][0], convert_to_decibel(h["anchor"][1]))
        #h["target"] = (h["target"][0], convert_to_decibel(h["target"][1]))
        line, = ax.plot([], [], color="lime", linewidth=1)
        hash_lines.append(line)

    def update(frame_time):
        playback_line.set_xdata([times[frame_time], times[frame_time]])
        #playback_line.set_xdata(frame_time)
        # Update hash lines
        for line, h in zip(hash_lines, hashes):
            if h["t_start"] <= frame_time <= h["t_end"]:
                line.set_data(
                    [h["anchor"][0], h["target"][0]],
                    [h["anchor"][1], h["target"][1]]
                )
            else:
                line.set_data([], [])
        return [playback_line] + hash_lines
        #return (playback_line,)

    frames = len(times)
    fps = 30
    duration = times.max()

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 * hop_length / sr, blit=True)
    ##ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)

    video_path = "tmp.mp4"
    output_path = "spectrogram.mp4"
    ani.save(video_path, writer="ffmpeg", fps=fps)
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    video_with_audio = video_clip.set_audio(audio_clip)
    video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac')

    #cmd = [
        #"ffmpeg",
        #"-i", tmp_video,
        #"-i", audio_path,
        #"-c:v", "copy",
        #"-c:a", "aac",
        #"-shortest",
        #output_video
    #]
    #subprocess.run(cmd, check=True)

    # Clean up temp file
    os.remove(video_path)
    print("done")


def visualize_map_interactive(audio_path):
    audio, sr = preprocess_audio(audio_path)
    frequencies, times, magnitudes = compute_fft(audio, sr)
    magnitudes = convert_to_decibel(magnitudes)
    print(magnitudes.shape)

    constellation_map = find_peaks(frequencies, times, magnitudes)
    peak_times = [times[t] for t, f in constellation_map]
    peak_freqs = [f for t, f in constellation_map]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=magnitudes,
        x=times,
        y=frequencies,
        colorscale='Inferno',
        colorbar=dict(title='Magnitude (dB)'),
        zsmooth='best',
        name="Spectrogram"
    ))

    fig.add_trace(go.Scatter(
        x=peak_times,
        y=peak_freqs,
        mode='markers',
        marker=dict(size=7, color='white', symbol='square-open'),
        name='Constellation Map',
        visible=True
    ))

    # overlay constellation peaks with a toggleable checkbox
    fig.update_layout(
        title='Spectrogram with Constellation Map',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        yaxis=dict(
            range=[frequencies.min(), frequencies.max()],
            autorange=False  # Prevent auto-padding when scatter appears
        ),
        xaxis=dict(
            range=[times.min(), times.max()],
            autorange=False  # Optional: lock X-axis too
        ),
        margin=dict(t=40, b=40, l=60, r=40),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(label="Show Peaks",
                         method="update",
                         args=[{"visible": [True, True]}]),
                    dict(label="Hide Peaks",
                         method="update",
                         args=[{"visible": [True, False]}]),
                ],
                showactive=True,
                x=1.05,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    fig.show()


def visualize_scoring(sample_wav_path: str) -> None:
    scores, time_pair_bins = recognize_music(sample_wav_path, remove_sample=False)
    song_ids = [score[0] for score in scores[:min(len(scores), 5)]]
    if 20 not in song_ids:
        song_ids.append(20)
    for song_id in song_ids:
        time_pair_bin = time_pair_bins[song_id]
        song = retrieve_song(song_id)
        if song is None:
            continue
        deltaT_values = [sourceT - sampleT for (sourceT, sampleT) in time_pair_bin]
        hist, bin_edges = np.histogram(deltaT_values, bins=max(len(np.unique(deltaT_values)), 10))
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(f"{song['title']} by {song['artist']}", fontsize=16)

        # scatterplot
        sourceT_vals = [pair[0] for pair in time_pair_bin]
        sampleT_vals = [pair[1] for pair in time_pair_bin]
        axes[0].scatter(sourceT_vals, sampleT_vals, alpha=0.7)
        axes[0].set_xlabel('Source Time')
        axes[0].set_ylabel('Sample Time')
        axes[0].set_title('Scatterplot of matching hash locations')

        # histogram
        axes[1].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge')
        axes[1].set_xlabel('Offset t')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Histogram of differences of time offsets')

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Music recognition CLI")
    parser.add_argument('--constellation', action='store_true', help='Visualize constellation map overlaid on spectrogram')
    parser.add_argument('--scoring', action='store_true', help='Visualize the score for top five closest matches')
    parser.add_argument('--spectrogram', action='store_true', help='Visualize spectrogram as a video with song audio')
    args = parser.parse_args()

    if args.constellation:
        visualize_map_interactive("sacrifice.mp3")
        #visualize_map_interactive("sacrifice_smallcopy.wav")
        #visualize_map_interactive("sacrifice_sample.wav")
        return
    elif args.scoring:
        #visualize_scoring("sacrifice_copy.wav")
        #visualize_scoring("sacrifice_smallcopy.wav")
        visualize_scoring("sacrifice_sample.wav")
        return
    elif args.spectrogram:
        visualize_spectrogram_as_mp4()
        return

if __name__ == "__main__":
    main()