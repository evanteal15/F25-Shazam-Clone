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
  View,
  Modal,
  Alert,
  Platform,
} from "react-native";
import { setStatusBarNetworkActivityIndicatorVisible } from "expo-status-bar";

export default function Recorder() {
  const [audioInput, setAudioInput] = useState("device");
  const [uri, setUri] = useState<any>(null);
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

      // let typet: string;
      // if (Platform.OS === "ios") {
      //   typet = "audio/m4a";
      // } else if (Platform.OS === "android") {
      //   typet = "audio/caf";
      // } else {
      //   throw Error("Unsupported file type");
      // }

      // const audioData = {
      //   uri: uri,
      //   type: "audio/m4a",
      //   name: "audio/m4a",
      // };
      // console.log("Image data being sent:", audioData);

      // formData.append("audio", audioData as any);

      const responsee = await fetch(uri);
      const blob = await responsee.blob();

      const file = new File([blob], "audio.wav", { type: "audio/wav" });

      console.log("Audio file being sent:", file);

      // Append the actual file to FormData
      formData.append("audio", file);

      console.log("Sending POST request to server...");
      const response = await fetch("http://172.30.123.55:5003/predict", {
        method: "POST",
        body: formData,
        // headers: {
        //   "Content-Type": "multipart/form-data",
        // },
      });

      // console.log("Response status:", response.status);
      // const responseText = await response.text();
      // console.log("Raw response:", responseText);

      // const data = JSON.parse(responseText);
      // if (!data.song) {
      //   console.error("No song data in response:", data);
      //   return null;
      // }

      // setPredictedSong(data.song);
      // return predictedSong;
      return file;
    } catch (error) {
      console.error("Error in getPredictedSong:", error);
      return null;
    }
  }

  const handlePrediction = async () => {
    if (uri) {
      const song = await getPredictedSong();
      if (song) {
        setPredictedSong(song.name);
        setShowPrediction(true);
      }
    }
  };

  return (
    <>
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
        </View>
      )}
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#25292e",
  },
});
