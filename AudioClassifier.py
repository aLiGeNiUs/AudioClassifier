import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, ttk
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import threading
import os
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import noisereduce as nr
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageTk
import librosa
import time

# Language translations
translations = {
    "ar": {
        "title": "مصنف الصوت",
        "about": "عن البرنامج",
        "reset": "تصفير",
        "select_method": "اختر طريقة التصنيف:",
        "use_pretrained": "استخدام نموذج مُدرب مسبقًا",
        "train_new": "تدريب نموذج جديد",
        "upload_training_data": "تحميل بيانات التدريب",
        "start_training": "بدء التدريب",
        "save_model": "حفظ النموذج المُدرب",
        "record_audio": "تسجيل صوت",
        "stop_recording": "إيقاف التسجيل",
        "upload_file": "تحميل ملف صوتي",
        "classify_audio": "تصنيف الصوت",
        "save_results": "حفظ النتائج",
        "results": "النتائج:",
        "status_ready": "جاهز",
        "status_recording": "جاري التسجيل...",
        "status_processing": "جاري المعالجة...",
        "status_classifying": "جاري التصنيف...",
        "status_training": "جاري التدريب...",
        "status_epoch": "اكتمل العصر {}/{}",
        "status_training_done": "اكتمل التدريب",
        "status_classification_done": "اكتمل التصنيف",
        "status_reset": "تم التصفير وإعادة التحميل",
        "status_uploaded": "تم التحميل: {}",
        "status_model_loaded": "تم تحميل النموذج المُدرب مسبقًا",
        "status_training_data_uploaded": "تم تحميل بيانات التدريب",
        "status_model_saved": "تم حفظ النموذج المُدرب",
        "status_recording_saved": "تم حفظ التسجيل",
        "warning_no_audio": "يرجى تسجيل أو تحميل ملف صوتي أولاً!",
        "warning_no_training_data": "يرجى تحميل بيانات التدريب أولاً!",
        "warning_no_model": "لا يوجد نموذج للحفظ!",
        "error": "خطأ",
        "error_classification_failed": "فشل التصنيف: {}",
        "error_training_failed": "فشل التدريب: {}",
        "error_channels": "المتوقع 1 أو 2 قنوات، تم العثور على {} قنوات",
        "predicted_class": "الفئة المتوقعة: {}\n",
        "confidence": "الثقة: {:.2%}\n",
        "probabilities": "\nتوزيع الاحتمالات:\n",
        "prob_entry": "{}: {:.2%}\n",
        "audio_duration": "مدة الصوت: {:.2f} ثانية\n",
        "snr": "مستوى الضوضاء (SNR): {:.2f} ديسيبل\n",
        "noise_reduction": "إزالة الضوضاء",
        "language": "اللغة",
        "close": "إغلاق",
        "warning_no_results": "لا توجد نتائج لحفظها!",
        "status_results_saved": "تم حفظ النتائج",
        "training_info": "معلومات التدريب:\n",
        "epoch_info": "العصر {}/{}:\n",
        "loss_info": "الخسارة: {:.4f}\n",
        "samples_processed": "عدد العينات المعالجة: {}\n",
        "time_taken": "الوقت المستغرق: {:.2f} ثانية\n",
        "about_message": "تطبيق مصنف الأصوات  الأصدار 1.0.0.0\nالمطور: علي عاصف --- 2025"
    },
    "en": {
        "title": "Audio Classifier",
        "about": "About",
        "reset": "Reset",
        "select_method": "Select Classification Method:",
        "use_pretrained": "Use Pretrained Model",
        "train_new": "Train New Model",
        "upload_training_data": "Upload Training Data",
        "start_training": "Start Training",
        "save_model": "Save Trained Model",
        "record_audio": "Record Audio",
        "stop_recording": "Stop Recording",
        "upload_file": "Upload Audio File",
        "classify_audio": "Classify Audio",
        "save_results": "Save Results",
        "results": "Results:",
        "status_ready": "Ready",
        "status_recording": "Recording...",
        "status_processing": "Processing...",
        "status_classifying": "Classifying...",
        "status_training": "Training...",
        "status_epoch": "Epoch {}/{} Completed",
        "status_training_done": "Training Completed",
        "status_classification_done": "Classification Completed",
        "status_reset": "Reset and Reloaded",
        "status_uploaded": "Uploaded: {}",
        "status_model_loaded": "Pretrained Model Loaded",
        "status_training_data_uploaded": "Training Data Uploaded",
        "status_model_saved": "Trained Model Saved",
        "status_recording_saved": "Recording Saved",
        "warning_no_audio": "Please record or upload an audio file first!",
        "warning_no_training_data": "Please upload training data first!",
        "warning_no_model": "No model to save!",
        "error": "Error",
        "error_classification_failed": "Classification Failed: {}",
        "error_training_failed": "Training Failed: {}",
        "error_channels": "Expected 1 or 2 channels, found {} channels",
        "predicted_class": "Predicted Class: {}\n",
        "confidence": "Confidence: {:.2%}\n",
        "probabilities": "\nProbability Distribution:\n",
        "prob_entry": "{}: {:.2%}\n",
        "audio_duration": "Audio Duration: {:.2f} seconds\n",
        "snr": "Noise Level (SNR): {:.2f} dB\n",
        "noise_reduction": "Noise Reduction",
        "language": "Language",
        "close": "Close",
        "warning_no_results": "No results to save!",
        "status_results_saved": "Results Saved",
        "training_info": "Training Information:\n",
        "epoch_info": "Epoch {}/{}:\n",
        "loss_info": "Loss: {:.4f}\n",
        "samples_processed": "Samples Processed: {}\n",
        "time_taken": "Time Taken: {:.2f} seconds\n",
        "about_message": "Audio Classifier Application Version 1.0.0.0\nDeveloper: Ali Asif --- 2025"
    }
}

class AudioDataset(Dataset):
    def __init__(self, csv_file, feature_extractor, sample_rate=16000):
        self.data = pd.read_csv(csv_file)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.label_map = {"Happy": 0, "Sad": 1, "Angry": 2, "Neutral": 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]["audio_path"]
        label = self.label_map[self.data.iloc[idx]["label"]]
        
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        
        return {
            "input_values": inputs.input_values.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class AudioClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Classifier")
        self.root.geometry("800x600")
        
        # Audio settings
        self.sample_rate = 16000
        self.recording = False
        self.audio_data = []
        self.audio_path = None
        self.results_history = []
        
        # Class labels
        self.class_labels = ["Happy", "Sad", "Angry", "Neutral"]
        
        # Training settings
        self.use_pretrained = True
        self.model = None
        self.feature_extractor = None
        self.speech_detection_model = None
        self.speech_processor = None
        self.training_data_path = None
        
        # Noise reduction setting
        self.use_noise_reduction = tk.BooleanVar(value=True)
        
        # Language setting
        self.current_language = "ar"
        
        # Setup GUI
        self.create_widgets()
        self.load_models()

    def reshape_text(self, text):
        if self.current_language == "ar":
            reshaped_text = arabic_reshaper.reshape(text)
            return get_display(reshaped_text)
        return text

    def get_text(self, key, *args):
        text = translations[self.current_language][key]
        if args:
            return text.format(*args)
        return text

    def load_models(self):
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
            self.model.eval()
            
            # Load speech detection model
            self.speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.speech_detection_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")
            self.speech_detection_model.eval()
            
            self.status_label.config(text=self.reshape_text(self.get_text("status_model_loaded")))
        except Exception as e:
            messagebox.showerror(self.reshape_text(self.get_text("error")), f"Failed to load models: {str(e)}")
            self.status_label.config(text=self.reshape_text(self.get_text("error")))

    def reset_app(self):
        self.audio_path = None
        self.audio_data = []
        self.result_text.delete(1.0, tk.END)
        self.spectrogram_label.config(image='')
        if os.path.exists("recorded_audio.wav"):
            os.remove("recorded_audio.wav")
        self.load_models()
        self.status_label.config(text=self.reshape_text(self.get_text("status_reset")))

    def show_about(self):
        about_window = Toplevel(self.root)
        about_window.title(self.reshape_text(self.get_text("about")))
        about_window.geometry("300x100")
        about_window.resizable(False, False)
        
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        about_window.geometry(f"+{root_x + 50}+{root_y + 50}")
        
        about_message = self.reshape_text(self.get_text("about_message"))
        tk.Label(about_window, text=about_message, font=("Arial", 12), justify="center").pack(expand=True)
        
        tk.Button(about_window, text=self.reshape_text(self.get_text("close")), command=about_window.destroy).pack(pady=5)

    def change_language(self, lang):
        self.current_language = lang
        self.update_ui_texts()

    def update_ui_texts(self):
        self.root.title("Audio Classifier")
        self.title_label.config(text=self.reshape_text(self.get_text("title")))
        self.about_btn.config(text=self.reshape_text(self.get_text("about")))
        self.reset_btn.config(text=self.reshape_text(self.get_text("reset")))
        self.select_method_label.config(text=self.reshape_text(self.get_text("select_method")))
        self.use_pretrained_radio.config(text=self.reshape_text(self.get_text("use_pretrained")))
        self.train_new_radio.config(text=self.reshape_text(self.get_text("train_new")))
        self.upload_training_btn.config(text=self.reshape_text(self.get_text("upload_training_data")))
        self.start_training_btn.config(text=self.reshape_text(self.get_text("start_training")))
        self.save_model_btn.config(text=self.reshape_text(self.get_text("save_model")))
        self.record_btn.config(text=self.reshape_text(self.get_text("record_audio")))
        self.upload_btn.config(text=self.reshape_text(self.get_text("upload_file")))
        self.classify_btn.config(text=self.reshape_text(self.get_text("classify_audio")))
        self.save_results_btn.config(text=self.reshape_text(self.get_text("save_results")))
        self.results_label.config(text=self.reshape_text(self.get_text("results")))
        self.status_label.config(text=self.reshape_text(self.get_text("status_ready")))
        self.noise_reduction_check.config(text=self.reshape_text(self.get_text("noise_reduction")))
        self.language_label.config(text=self.reshape_text(self.get_text("language")))

    def create_widgets(self):
        # Language selection
        language_frame = tk.Frame(self.root, bg="#f0f0f0")
        language_frame.pack(fill="x")
        self.language_label = tk.Label(language_frame, text=self.reshape_text(self.get_text("language")), font=("Arial", 10), bg="#f0f0f0")
        self.language_label.pack(side="left", padx=5)
        language_options = ["ar", "en"]
        self.language_var = tk.StringVar(value="ar")
        language_menu = ttk.Combobox(language_frame, textvariable=self.language_var, values=language_options, state="readonly", width=10)
        language_menu.pack(side="left", padx=5)
        language_menu.bind("<<ComboboxSelected>>", lambda event: self.change_language(self.language_var.get()))
        
        # Noise reduction option
        self.noise_reduction_check = tk.Checkbutton(language_frame, text=self.reshape_text(self.get_text("noise_reduction")), variable=self.use_noise_reduction, bg="#f0f0f0")
        self.noise_reduction_check.pack(side="right", padx=5)
        
        # About and Reset buttons
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(fill="x", pady=5)
        self.about_btn = tk.Button(btn_frame, text=self.reshape_text(self.get_text("about")), command=self.show_about, width=15)
        self.about_btn.pack(side="left", padx=5)
        self.reset_btn = tk.Button(btn_frame, text=self.reshape_text(self.get_text("reset")), command=self.reset_app, width=15)
        self.reset_btn.pack(side="left", padx=5)
        
        # Title
        self.title_label = tk.Label(self.root, text=self.reshape_text(self.get_text("title")), font=("Arial", 18, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=20)
        
        # Model selection frame
        model_frame = tk.Frame(self.root, bg="#f0f0f0")
        model_frame.pack(pady=10)
        self.select_method_label = tk.Label(model_frame, text=self.reshape_text(self.get_text("select_method")), font=("Arial", 12), bg="#f0f0f0")
        self.select_method_label.grid(row=0, column=0, columnspan=2)
        self.model_var = tk.BooleanVar(value=True)
        self.use_pretrained_radio = tk.Radiobutton(model_frame, text=self.reshape_text(self.get_text("use_pretrained")), variable=self.model_var, value=True, command=self.toggle_model, bg="#f0f0f0")
        self.use_pretrained_radio.grid(row=1, column=0, padx=5)
        self.train_new_radio = tk.Radiobutton(model_frame, text=self.reshape_text(self.get_text("train_new")), variable=self.model_var, value=False, command=self.toggle_model, bg="#f0f0f0")
        self.train_new_radio.grid(row=1, column=1, padx=5)
        
        # Training data upload (hidden by default)
        self.train_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.upload_training_btn = tk.Button(self.train_frame, text=self.reshape_text(self.get_text("upload_training_data")), command=self.upload_training_data)
        self.upload_training_btn.pack(pady=5)
        self.start_training_btn = tk.Button(self.train_frame, text=self.reshape_text(self.get_text("start_training")), command=self.train_model)
        self.start_training_btn.pack(pady=5)
        self.save_model_btn = tk.Button(self.train_frame, text=self.reshape_text(self.get_text("save_model")), command=self.save_model)
        self.save_model_btn.pack(pady=5)
        
        # Recording controls frame
        record_frame = tk.Frame(self.root, bg="#f0f0f0")
        record_frame.pack(pady=10)
        self.record_btn = tk.Button(record_frame, text=self.reshape_text(self.get_text("record_audio")), command=self.toggle_recording, width=15)
        self.record_btn.grid(row=0, column=0, padx=5)
        self.upload_btn = tk.Button(record_frame, text=self.reshape_text(self.get_text("upload_file")), command=self.upload_file, width=15)
        self.upload_btn.grid(row=0, column=1, padx=5)
        
        # Classify and Save Results buttons
        action_frame = tk.Frame(self.root, bg="#f0f0f0")
        action_frame.pack(pady=10)
        self.classify_btn = tk.Button(action_frame, text=self.reshape_text(self.get_text("classify_audio")), command=self.classify_audio, width=15)
        self.classify_btn.pack(side="left", padx=5)
        self.save_results_btn = tk.Button(action_frame, text=self.reshape_text(self.get_text("save_results")), command=self.save_results, width=15)
        self.save_results_btn.pack(side="left", padx=5)
        
        # Results and Spectrogram in a horizontal frame
        results_frame = tk.Frame(self.root, bg="#f0f0f0")
        results_frame.pack(pady=5, fill="both", expand=True)
        
        # Spectrogram on the left
        self.spectrogram_label = tk.Label(results_frame, bg="#f0f0f0")
        self.spectrogram_label.pack(side="left", padx=5)
        
        # Results on the right
        results_inner_frame = tk.Frame(results_frame, bg="#f0f0f0")
        results_inner_frame.pack(side="left", fill="both", expand=True)
        self.results_label = tk.Label(results_inner_frame, text=self.reshape_text(self.get_text("results")), font=("Arial", 12), bg="#f0f0f0")
        self.results_label.pack(anchor="w")
        self.result_text = tk.Text(results_inner_frame, height=10, width=40)
        self.result_text.pack(pady=5, fill="both", expand=True)
        
        # Status bar
        self.status_label = tk.Label(self.root, text=self.reshape_text(self.get_text("status_ready")), font=("Arial", 10), relief="sunken", anchor="w", bg="#d0d0d0")
        self.status_label.pack(fill="x", padx=10, pady=10)

    def toggle_model(self):
        self.use_pretrained = self.model_var.get()
        if self.use_pretrained:
            self.train_frame.pack_forget()
            self.load_models()
        else:
            self.train_frame.pack(pady=10)
            self.status_label.config(text=self.reshape_text(self.get_text("status_training_data_uploaded")))

    def upload_training_data(self):
        self.training_data_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")]
        )
        if self.training_data_path:
            self.status_label.config(text=self.reshape_text(self.get_text("status_training_data_uploaded")))

    def train_model(self):
        if not self.training_data_path:
            messagebox.showwarning(self.reshape_text(self.get_text("warning_no_training_data")), self.reshape_text(self.get_text("warning_no_training_data")))
            return
        
        try:
            self.status_label.config(text=self.reshape_text(self.get_text("status_training")))
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, self.reshape_text(self.get_text("training_info")))
            
            dataset = AudioDataset(self.training_data_path, self.feature_extractor, self.sample_rate)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
            criterion = nn.CrossEntropyLoss()
            
            num_epochs = 3
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                epoch_loss = 0.0
                samples_processed = 0
                
                for batch in dataloader:
                    input_values = batch["input_values"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_values, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    samples_processed += input_values.size(0)
                
                epoch_time = time.time() - epoch_start_time
                avg_loss = epoch_loss / len(dataloader)
                
                self.result_text.insert(tk.END, self.reshape_text(self.get_text("epoch_info", epoch+1, num_epochs)))
                self.result_text.insert(tk.END, self.reshape_text(self.get_text("loss_info", avg_loss)))
                self.result_text.insert(tk.END, self.reshape_text(self.get_text("samples_processed", samples_processed)))
                self.result_text.insert(tk.END, self.reshape_text(self.get_text("time_taken", epoch_time)))
                self.result_text.insert(tk.END, "\n")
                self.result_text.see(tk.END)
                
                self.status_label.config(text=self.reshape_text(self.get_text("status_epoch", epoch+1, num_epochs)))
            
            self.model.eval()
            self.status_label.config(text=self.reshape_text(self.get_text("status_training_done")))

        except Exception as e:
            messagebox.showerror(self.reshape_text(self.get_text("error")), self.reshape_text(self.get_text("error_training_failed", str(e))))
            self.status_label.config(text=self.reshape_text(self.get_text("error")))

    def save_model(self):
        if self.model is None or self.feature_extractor is None:
            messagebox.showwarning(self.reshape_text(self.get_text("warning_no_model")), self.reshape_text(self.get_text("warning_no_model")))
            return
        
        save_dir = filedialog.askdirectory()
        if save_dir:
            self.model.save_pretrained(save_dir)
            self.feature_extractor.save_pretrained(save_dir)
            self.status_label.config(text=self.reshape_text(self.get_text("status_model_saved")))

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_btn.config(text=self.reshape_text(self.get_text("stop_recording")))
            self.status_label.config(text=self.reshape_text(self.get_text("status_recording")))
            self.audio_data = []
            
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.start()
        else:
            self.recording = False
            self.record_btn.config(text=self.reshape_text(self.get_text("record_audio")))
            self.status_label.config(text=self.reshape_text(self.get_text("status_processing")))

    def record_audio(self):
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())
                
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
                while self.recording:
                    sd.sleep(100)
        
            if self.audio_data:
                audio_array = np.concatenate(self.audio_data, axis=0)
                self.audio_path = "recorded_audio.wav"
                wavfile.write(self.audio_path, self.sample_rate, audio_array)
                self.status_label.config(text=self.reshape_text(self.get_text("status_recording_saved")))
        except Exception as e:
            messagebox.showerror(self.reshape_text(self.get_text("error")), f"Recording failed: {str(e)}")
            self.status_label.config(text=self.reshape_text(self.get_text("error")))

    def upload_file(self):
        self.audio_path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3")]
        )
        if self.audio_path:
            self.status_label.config(text=self.reshape_text(self.get_text("status_uploaded", self.audio_path.split('/')[-1])))
            self.result_text.delete(1.0, tk.END)
            self.spectrogram_label.config(image='')

    def normalize_audio(self, waveform):
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.std() + 1e-8)
        return waveform

    def calculate_snr(self, waveform):
        waveform_np = waveform.numpy()
        signal_power = np.mean(waveform_np ** 2)
        noise_power = np.var(waveform_np - np.mean(waveform_np))
        if noise_power == 0:
            return float('inf')
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def display_spectrogram(self, waveform, sample_rate):
        try:
            plt.figure(figsize=(6, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform.numpy().squeeze())), ref=np.max)
            librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Spectrogram")
            
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            image = Image.open(buf)
            image = image.resize((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.spectrogram_label.config(image=photo)
            self.spectrogram_label.image = photo
            plt.close()
        except Exception as e:
            messagebox.showerror(self.reshape_text(self.get_text("error")), f"Failed to display spectrogram: {str(e)}")

    def detect_speech(self, waveform, sample_rate):
        try:
            inputs = self.speech_processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.speech_detection_model(**inputs)
                logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            return predicted_id == 1  # Assuming 1 means speech
        except Exception as e:
            messagebox.showerror(self.reshape_text(self.get_text("error")), f"Speech detection failed: {str(e)}")
            return False

    def classify_audio(self):
        if not self.audio_path:
            messagebox.showwarning(self.reshape_text(self.get_text("warning_no_audio")), self.reshape_text(self.get_text("warning_no_audio")))
            return
        
        try:
            self.status_label.config(text=self.reshape_text(self.get_text("status_classifying")))
            
            waveform, sample_rate = torchaudio.load(self.audio_path)
            
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif waveform.shape[0] != 1:
                raise ValueError(self.reshape_text(self.get_text("error_channels", waveform.shape[0])))
            
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Noise reduction
            if self.use_noise_reduction.get():
                waveform_np = waveform.numpy()
                waveform_np = nr.reduce_noise(y=waveform_np, sr=sample_rate)
                if waveform_np.ndim == 1:
                    waveform_np = waveform_np[np.newaxis, :]
                elif waveform_np.ndim > 2:
                    waveform_np = waveform_np.squeeze()
                    waveform_np = waveform_np[np.newaxis, :]
                waveform = torch.tensor(waveform_np)
            
            waveform = self.normalize_audio(waveform)
            
            # Speech detection
            is_speech = self.detect_speech(waveform, self.sample_rate)
            if not is_speech:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "No speech detected in the audio.\n")
                self.status_label.config(text=self.reshape_text(self.get_text("status_classification_done")))
                self.audio_path = None
                return
            
            # Calculate audio duration and SNR
            duration = waveform.shape[1] / sample_rate
            snr = self.calculate_snr(waveform)
            
            # Display spectrogram
            self.display_spectrogram(waveform, sample_rate)
            
            # Classify emotion
            inputs = self.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
            predicted_label = self.class_labels[predicted_id]
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, self.reshape_text(self.get_text("audio_duration", duration)))
            self.result_text.insert(tk.END, self.reshape_text(self.get_text("snr", snr)))
            self.result_text.insert(tk.END, self.reshape_text(self.get_text("predicted_class", predicted_label)))
            self.result_text.insert(tk.END, self.reshape_text(self.get_text("confidence", confidence)))
            self.result_text.insert(tk.END, self.reshape_text(self.get_text("probabilities")))
            for i, (label, prob) in enumerate(zip(self.class_labels, probabilities[0])):
                self.result_text.insert(tk.END, self.reshape_text(self.get_text("prob_entry", label, prob)))
            
            # Save result to history
            self.results_history.append({
                "audio_path": self.audio_path,
                "predicted_class": predicted_label,
                "confidence": confidence,
                "duration": duration,
                "snr": snr
            })
            
            self.status_label.config(text=self.reshape_text(self.get_text("status_classification_done")))
            
            # Clear audio_path after classification
            self.audio_path = None

        except Exception as e:
            messagebox.showerror(self.reshape_text(self.get_text("error")), self.reshape_text(self.get_text("error_classification_failed", str(e))))
            self.status_label.config(text=self.reshape_text(self.get_text("error")))

    def save_results(self):
        if not self.results_history:
            messagebox.showwarning(self.reshape_text(self.get_text("warning_no_results")), self.reshape_text(self.get_text("warning_no_results")))
            return
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if save_path:
            df = pd.DataFrame(self.results_history)
            df.to_csv(save_path, index=False)
            self.status_label.config(text=self.reshape_text(self.get_text("status_results_saved")))

def main():
    root = tk.Tk()
    app = AudioClassifierApp(root)
    root.configure(bg="#f0f0f0")
    root.mainloop()

if __name__ == "__main__":
    main()
