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
} from "react-native";
import { SafeAreaView, SafeAreaProvider } from "react-native-safe-area-context";
import { WebView } from "react-native-webview";
import { setStatusBarNetworkActivityIndicatorVisible } from "expo-status-bar";

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

      const response = await fetch("http://192.168.1.170:5003/predict", {
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
      await fetch("http://192.168.1.170:5003/add", {
        method: "POST",
        body: JSON.stringify({ youtube_url: song_url }),
        headers: {
          "Content-Type": "application/json",
        },
      });
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
        <SafeAreaView>
          <View style={styles.container}>
            <Button
              title={
                recorderState.isRecording ? "Stop Recording" : "Start Recording"
              }
              onPress={recorderState.isRecording ? stopRecording : record}
            />
          </View>
          <View style={styles.container}>
            <Button
              title="Play Recording"
              onPress={async () => {
                // console.log("Loading sound..");
                const { sound } = await Audio.Sound.createAsync(
                  { uri: uri },
                  { shouldPlay: true }
                );

                // sound.setOnPlaybackStatusUpdate((status) => {
                //   setIsPlaying(status.isPlaying);

                //   if (status.didJustFinish) {
                //     setIsPlaying(false); // Reset playing status when finished
                //   }
                // });

                // console.log("Playing sound..");
                await sound.playAsync();

                // soundRef.current = sound;
              }}
            />
          </View>
          <View style={styles.container}>
            <TouchableOpacity onPress={handlePrediction}>
              <Text>Predict Song</Text>
            </TouchableOpacity>
          </View>
          {showPrediction && uri && (
            <View style={styles.container}>
              <Text>Predicted Song: {predictedSong}</Text>
              <Text>Predicted Confidence: {predictedConfidence}</Text>
              <Text>Predicted URL: {predictedUrl}</Text>
              {/* <WebView
            style={{ flex: 1 }}
            javaScriptEnabled={true}
            source={{
              uri: "",
            }}
          /> */}
            </View>
          )}

          {/* <form
        onSubmit={(e) => {
          e.preventDefault();
          addSongToDatabase(text);
          setText("");
        }}
      >
        <TextInput value={text} onChangeText={(value) => setText(value)} />
      </form> */}
          <View style={styles.container}>
            <TextInput
              value={text}
              onChangeText={setText}
              placeholder="Type your song..."
              // style={styles.input}
              // Optionally submit on return key:
              onSubmitEditing={(e) => {
                e.preventDefault();
                addSongToDatabase(text);
                setText("");
              }}
              returnKeyType="done"
            />
            <Button
              title="Add Song"
              onPress={(e) => {
                e.preventDefault();
                addSongToDatabase(text);
                setText("");
              }}
            />
          </View>
        </SafeAreaView>
      </SafeAreaProvider>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#25292e",
  },
});
