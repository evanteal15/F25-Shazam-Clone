from cm_helper import create_samples, add_noise
from DBcontrol import init_db, connect

from parameters import set_parameters

microphone_sample_list = [
    ("Plastic Beach", "Gorillaz", "audio_samples/plastic_beach_microphone_recording.wav")
    # ...
]

microphone_sample_list = [
    {"title": v[0], "artist": v[1], "audio_path": v[2]} 
    for v in microphone_sample_list
    ]

def augment_samples(sr, noise_weight):
    con = connect()
    cur = con.cursor()
    samples = []
    # TODO for members: Write this SQL query
    for sample in microphone_sample_list:
        cur.execute(
            "SELECT id AS song_id "
            "FROM songs "
            "WHERE title ILIKE %?% "  # could use LIKE, could use =
            "AND artist ILIKE %?% ",
            (sample["title"], sample["artist"])
        )

        song_id = cur.fetchall()[0]["song_id"]
        sample_slices = create_samples(sample["audio_path"], sr, n_samples = 20)
        sample_slices_noisy = [add_noise(audio, noise_weight) for audio in sample_slices]
        samples.extend([
            {"song_id": song_id,
             "mic_clean": s[0],
             "mic_noise": s[1]
             } for s in zip(sample_slices, sample_slices_noisy)
        ])
    return samples



def perform_recognition_test():
    """
    """
    #init_db()
    sr = 11025
    init_db(n_songs=2)
    samples = augment_samples(sr=sr, noise_weight=0.3)
    # ...




def compute_performance_metrics():
    pass


def run_grid_search():
    # define search space programmatically
    pass


if __name__ == "__main__":
    run_grid_search()