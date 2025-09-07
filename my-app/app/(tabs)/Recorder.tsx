import React, { useEffect, useState, useRef } from "react";
import {
  useAudioRecorder,
  AudioModule,
  useAudioRecorderState,
  setAudioModeAsync,
  RecordingPresets,
  RecordingStatus,
  AudioRecorder,
  useAudioPlayer,
} from "expo-audio";
import { Audio } from "expo-av";
import {
  Button,
  StyleSheet,
  Text,
  TouchableOpacity,
  TextInput,
  View,
  Alert,
  Platform,
  KeyboardAvoidingView,
  ActivityIndicator,
} from "react-native";
import YoutubePlayer from "react-native-youtube-iframe";
import { SafeAreaView, SafeAreaProvider } from "react-native-safe-area-context";
import { WebView } from "react-native-webview";
import { setStatusBarNetworkActivityIndicatorVisible } from "expo-status-bar";

function extractYouTubeVideoId(url: string | null): string {
  if (!url) return "";
  const match = url.match(
    /(?:youtube\.com\/.*v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/
  );
  return match ? match[1] : url;
}

export default function Recorder() {
  const [audioInput, setAudioInput] = useState("device");
  const [uri, setUri] = useState<any>(null);
  const [text, setText] = React.useState("");
  const audioRecorder = useAudioRecorder({
    extension: ".m4a",
    numberOfChannels: 2,
    sampleRate: 44100,
    bitRate: 128000,
    android: {
      extension: ".wav",
      outputFormat: "mpeg4",
      audioEncoder: "aac",
      sampleRate: 44100,
    },
    ios: {
      extension: ".wav",
      audioQuality: 2,
      sampleRate: 44100,
      linearPCMBitDepth: 16,
      linearPCMIsBigEndian: false,
      linearPCMIsFloat: false,
    },
    web: {
      mimeType: "audio/wav",
    },
  });
  const recorderState = useAudioRecorderState(audioRecorder);
  const [predictedSong, setPredictedSong] = useState<string | null>(null);
  const [predictedConfidence, setPredictedConfidence] = useState<number | null>(
    null
  );
  const [predictedUrl, setPredictedUrl] = useState<string | null>(null);
  const [showPrediction, setShowPrediction] = useState(false);
  const [addingSong, setAddingSong] = useState(false);
  // const audioPlayer = useAudioPlayer();
  // const [audioPlayer, setAudioPlayer] = useState(null);
  //   const [permission, requestPermission] = useAudPermissions();

  const record = async () => {
    await audioRecorder.prepareToRecordAsync();
    audioRecorder.record();
  };

  const stopRecording = async () => {
    await audioRecorder.stop();
    setUri(audioRecorder.uri);
  };

  useEffect(() => {
    (async () => {
      const status = await AudioModule.getRecordingPermissionsAsync();
      if (!status.granted) {
        Alert.alert(
          "Permission to access microphone must be given to use the app"
        );
      }

      setAudioModeAsync({
        playsInSilentMode: true,
        allowsRecording: true,
      });
    })();
  }, []);

  async function getPredictedSong() {
    if (!uri) {
      console.error("No recording available");
      return;
    }

    try {
      console.log("Recording URI:", uri);

      const formData = new FormData();

      const responsee = await fetch(uri);
      const blob = await responsee.blob();

      const file = new File([blob], "audio.wav", { type: "audio/wav" });

      console.log("Audio file being sent:", file);

      // Append the actual file to FormData
      formData.append("audio", file);

      console.log("Sending POST request to server...");
      // const response = await fetch("http://172.30.123.55:5003/predict", {
      //   method: "POST",
      //   body: formData,
      //   // headers: {
      //   //   "Content-Type": "multipart/form-data",
      //   // },
      // });

      const response = await fetch("http://35.2.57.219:5003/predict", {
        method: "POST",
        body: formData,
        // headers: {
        //   "Content-Type": "multipart/form-data",
        // },
      });

      // setPredictedSong(data.song);
      // return predictedSong;
      return response;
    } catch (error) {
      console.error("Error in getPredictedSong:", error);
      return null;
    }
  }

  async function addSongToDatabase(song_url: any) {
    try {
      console.log("Adding song to database:", song_url);
      // await fetch("http://172.30.123.55:5003/add", {
      //   method: "POST",
      //   body: JSON.stringify({ youtube_url: song_url }),
      //   headers: {
      //     "Content-Type": "application/json",
      //   },
      // });
      setAddingSong(true);
      await fetch("http://35.2.57.219:5003/add", {
        method: "POST",
        body: JSON.stringify({ youtube_url: song_url }),
        headers: {
          "Content-Type": "application/json",
        },
      });
      setAddingSong(false);
      Alert.alert("Song added to database!");
    } catch (error) {
      console.error("Error adding song to database:", error);
    }
  }

  const handlePrediction = async () => {
    if (uri) {
      const response = await getPredictedSong();
      if (response) {
        const data = await response.json();
        setPredictedSong(data.best);
        setPredictedConfidence(data.confidence);
        setPredictedUrl(data.urls);
        setShowPrediction(true);
      }
    }
  };

  return (
    <>
      <SafeAreaProvider>
        <SafeAreaView style={styles.container}>
          <View style={{ width: 150, marginBottom: 20, marginTop: 50 }}>
            <Button
              title={
                recorderState.isRecording ? "Stop Recording" : "Start Recording"
              }
              onPress={recorderState.isRecording ? stopRecording : record}
            />
          </View>
          <View style={{ width: 150, marginBottom: 20, marginTop: 20 }}>
            <Button
              title="Play Recording"
              onPress={async () => {
                // console.log("Loading sound..");
                const { sound } = await Audio.Sound.createAsync(
                  { uri: uri },
                  { shouldPlay: true }
                );

                // console.log("Playing sound..");
                await sound.playAsync();
              }}
            />
          </View>
          <View style={{ width: 150, marginBottom: 10, marginTop: 20 }}>
            <Button title="Predict Song" onPress={handlePrediction} />
          </View>
          {predictedSong && (
            <View style={{ alignItems: "center", marginTop: 20 }}>
              <Text style={{ fontSize: 15, marginBottom: 20 }}>
                Here is the most likely song from the recorded snippet!
              </Text>
              {/* <Text>Predicted URL: {predictedUrl}</Text> */}
              {/* <YouTube
                    videoId={predictedUrl} // Extract this from your YouTube URL
                    opts={opts}
                    onReady={(event) => event.target.pauseVideo()}
                  /> */}
              <YoutubePlayer
                height={180}
                scale={4}
                play={false}
                videoId={extractYouTubeVideoId(predictedUrl)}
                style={{ alignSelf: "stretch", marginTop: 20 }}
              />
              <Text>If this does not look correct:</Text>
              <Text>1: Please try again with a new clip.</Text>
              <Text>2: Add the song to our database below.</Text>
            </View>
          )}

          <KeyboardAvoidingView
            behavior={Platform.OS === "ios" ? "padding" : "height"} // "padding" works best on iOS
            keyboardVerticalOffset={0} // adjust if you have headers/navbars
          >
            <TextInput
              style={{
                width: "100%",
                alignSelf: "center",
                marginBottom: 10,
                marginTop: 20,
                backgroundColor: "#e0e0e0",
                padding: 10,
                borderRadius: 5,
              }}
              value={text}
              onChangeText={setText}
              placeholder="Enter YouTube URL here..."
              // style={styles.input}
              // Optionally submit on return key:
              onSubmitEditing={(e) => {
                e.preventDefault();
                addSongToDatabase(text);
                setText("");
              }}
              returnKeyType="done"
            />
          </KeyboardAvoidingView>
          {addingSong && (
            <View style={{ width: 150, marginBottom: 20, marginTop: 10 }}>
              <ActivityIndicator size="small" color="#0000ff" />
            </View>
          )}
          {!addingSong && (
            <View style={{ width: 150, marginBottom: 20, marginTop: 10 }}>
              <Button
                title="Add Song"
                onPress={(e) => {
                  e.preventDefault();
                  addSongToDatabase(text);
                  setText("");
                }}
              />
            </View>
          )}
        </SafeAreaView>
      </SafeAreaProvider>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 20,
    backgroundColor: "#ffffffc5",
  },
});
