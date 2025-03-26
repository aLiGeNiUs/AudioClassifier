# AudioClassifier

س
Audio Classifier Application Guide
1. Introduction
The Audio Classifier Application is a tool designed to record, upload, and classify audio files based on emotions (Happy, Sad, Angry, Neutral). It uses a pre-trained Wav2Vec2 model for emotion classification and allows users to train a new model if needed. The application supports both English and Arabic languages for the user interface.
2. System Requirements
To run the Audio Classifier Application, ensure your system meets the following requirements:

    Operating System: Windows, macOS, or Linux.
    Python Version: Python 3.8 or higher.
    Hardware:
        At least 4 GB of RAM (8 GB recommended for training models).
        A microphone for recording audio.
    Internet Connection: Required for the initial installation of dependencies.
    Disk Space: At least 2 GB of free space for installing dependencies and saving models.

3. Installation
Step 1: Install Python

    Download and install Python 3.8 or higher from the official website: python.org.
    During installation, ensure you check the box to "Add Python to PATH" to make Python accessible from the command line.

Step 2: Set Up a Virtual Environment (Optional but Recommended)

    Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux).
    Navigate to the directory where you want to store the application:

    cd path/to/your/directory

    Create a virtual environment:

    python -m venv audio_classifier_env

    Activate the virtual environment:
        On Windows:

        audio_classifier_env\Scripts\activate

        On macOS/Linux:

        source audio_classifier_env/bin/activate

Step 3: Install Required Libraries

    With the virtual environment activated, install the required Python libraries using pip. Run the following command:

    pip install torch torchaudio transformers sounddevice numpy scipy pandas noisereduce matplotlib pillow librosa arabic-reshaper python-bidi

        Note: If you encounter issues installing torch or torchaudio, ensure you install the correct version compatible with your system. Visit pytorch.org for specific installation commands.
        If you are using a GPU, ensure you install the CUDA-enabled version of PyTorch.

Step 4: Download the Application Code

    Copy the entire code provided (the Python script named audio_classifier.py).
    Save it in your project directory as audio_classifier.py.

4. Running the Application

    Open a terminal and navigate to the directory containing audio_classifier.py:

    cd path/to/your/directory

    If you are using a virtual environment, ensure it is activated (see Step 2 in the Installation section).
    Run the application:

    python audio_classifier.py

    The application window will open, displaying the Audio Classifier interface.

5. Using the Application
Interface Overview

    Language Selection: At the top left, you can switch between English and Arabic.
    Noise Reduction Option: At the top right, you can enable/disable noise reduction for audio processing.
    About and Reset Buttons: At the top, you can access the "About" window or reset the application.
    Classification Method: Choose between using a pre-trained model or training a new model.
    Recording Controls: Buttons to record audio or upload an audio file.
    Action Buttons: Buttons to classify audio and save results.
    Results and Spectrogram: Displays the classification results and spectrogram of the audio.

Step 1: Select Language

    Use the dropdown menu at the top left to select your preferred language (English or Arabic).

Step 2: Choose Classification Method

    Use Pretrained Model: Select this option to use the default Wav2Vec2 model for classification (recommended for most users).
    Train New Model: Select this option if you want to train a new model. You will need to:
        Click "Upload Training Data" to select a CSV file containing paths to audio files and their labels (format: audio_path,label).
        Click "Start Training" to train the model (this may take some time depending on your hardware).
        Click "Save Trained Model" to save the trained model to a directory.

Step 3: Record or Upload Audio

    Record Audio:
        Click the "Record Audio" button to start recording.
        Speak into your microphone.
        Click the button again (now labeled "Stop Recording") to stop.
    Upload Audio:
        Click the "Upload Audio File" button.
        Select a .wav or .mp3 file from your computer.

Step 4: Classify Audio

    After recording or uploading an audio file, click the "Classify Audio" button.
    The application will process the audio and display:
        A spectrogram of the audio on the left.
        Classification results on the right, including:
            Audio duration.
            Signal-to-Noise Ratio (SNR).
            Predicted emotion (Happy, Sad, Angry, Neutral).
            Confidence score.
            Probability distribution for all emotions.

Step 5: Save Results

    After classification, click the "Save Results" button.
    Choose a location to save the results as a CSV file.
    The CSV file will contain the audio path, predicted class, confidence, duration, and SNR.

Step 6: Reset the Application (Optional)

    Click the "Reset" button at the top to clear all data and reload the pre-trained model.

Step 7: View About Information

    Click the "About" button to view information about the application, including the version and developer details.

6. Troubleshooting

    Application Fails to Start:
        Ensure all required libraries are installed correctly.
        Check if Python is added to your system PATH.
        Verify that your Python version is 3.8 or higher.
    Recording Issues:
        Ensure your microphone is connected and working.
        Check if the sounddevice library is installed correctly.
    Model Training Fails:
        Ensure your training data CSV file is formatted correctly.
        Check if you have enough memory (RAM) for training.
    Slow Performance:
        If training or classification is slow, consider using a GPU with CUDA support.
        Reduce the batch size in the train_model function if training is too slow.

7. Additional Notes

    The application automatically saves recorded audio as recorded_audio.wav in the project directory. This file is deleted when you reset the application.
    Training a new model requires a dataset in CSV format with two columns: audio_path (path to the audio file) and label (one of: Happy, Sad, Angry, Neutral).
    The application supports .wav and .mp3 audio files for upload.

دليل تطبيق مصنف الصوت
1. مقدمة
تطبيق مصنف الصوت هو أداة مصممة لتسجيل وتحميل وتصنيف الملفات الصوتية بناءً على المشاعر (سعيد، حزين، غاضب، محايد). يستخدم التطبيق نموذج Wav2Vec2 المدرب مسبقًا لتصنيف المشاعر، ويسمح للمستخدمين بتدريب نموذج جديد إذا لزم الأمر. يدعم التطبيق اللغتين العربية والإنجليزية لواجهة المستخدم.
2. متطلبات النظام
لتشغيل تطبيق مصنف الصوت، تأكد من أن نظامك يلبي المتطلبات التالية:

    نظام التشغيل: ويندوز، ماك أو إس، أو لينكس.
    إصدار بايثون: بايثون 3.8 أو أعلى.
    الأجهزة:
        ذاكرة وصول عشوائي (RAM) لا تقل عن 4 جيجابايت (يُوصى بـ 8 جيجابايت لتدريب النماذج).
        ميكروفون لتسجيل الصوت.
    اتصال بالإنترنت: مطلوب لتثبيت التبعيات لأول مرة.
    مساحة التخزين: ما لا يقل عن 2 جيجابايت من المساحة الحرة لتثبيت التبعيات وحفظ النماذج.

3. التثبيت
الخطوة 1: تثبيت بايثون

    قم بتنزيل وتثبيت بايثون 3.8 أو أعلى من الموقع الرسمي: python.org.
    أثناء التثبيت، تأكد من تحديد خيار "Add Python to PATH" لجعل بايثون متاحًا من سطر الأوامر.

الخطوة 2: إعداد بيئة افتراضية (اختياري ولكنه موصى به)

    افتح نافذة طرفية (Command Prompt في ويندوز، أو Terminal في ماك أو إس/لينكس).
    انتقل إلى المجلد الذي تريد تخزين التطبيق فيه:

    cd path/to/your/directory

    أنشئ بيئة افتراضية:

    python -m venv audio_classifier_env

    قم بتفعيل البيئة الافتراضية:
        في ويندوز:

        audio_classifier_env\Scripts\activate

        في ماك أو إس/لينكس:

        source audio_classifier_env/bin/activate

الخطوة 3: تثبيت المكتبات المطلوبة

    مع تفعيل البيئة الافتراضية، قم بتثبيت مكتبات بايثون المطلوبة باستخدام pip. نفّذ الأمر التالي:

    pip install torch torchaudio transformers sounddevice numpy scipy pandas noisereduce matplotlib pillow librosa arabic-reshaper python-bidi

        ملاحظة: إذا واجهت مشاكل أثناء تثبيت torch أو torchaudio، تأكد من تثبيت الإصدار المتوافق مع نظامك. قم بزيارة pytorch.org للحصول على أوامر تثبيت محددة.
        إذا كنت تستخدم وحدة معالجة رسوميات (GPU)، تأكد من تثبيت الإصدار المدعوم بـ CUDA من PyTorch.

الخطوة 4: تنزيل كود التطبيق

    انسخ كامل الكود المقدم (السكربت البايثون باسم audio_classifier.py).
    احفظه في مجلد المشروع الخاص بك باسم audio_classifier.py.

4. تشغيل التطبيق

    افتح نافذة طرفية وانتقل إلى المجلد الذي يحتوي على audio_classifier.py:

    cd path/to/your/directory

    إذا كنت تستخدم بيئة افتراضية، تأكد من تفعيلها (انظر الخطوة 2 في قسم التثبيت).
    شغّل التطبيق:

    python audio_classifier.py

    ستفتح نافذة التطبيق، وستظهر واجهة مصنف الصوت.

5. استخدام التطبيق
نظرة عامة على الواجهة

    اختيار اللغة: في أعلى اليسار، يمكنك التبديل بين اللغتين العربية والإنجليزية.
    خيار تقليل الضوضاء: في أعلى اليمين، يمكنك تفعيل/تعطيل تقليل الضوضاء لمعالجة الصوت.
    أزرار "عن البرنامج" و"تصفير": في الأعلى، يمكنك الوصول إلى نافذة "عن البرنامج" أو تصفير التطبيق.
    طريقة التصنيف: اختر بين استخدام نموذج مدرب مسبقًا أو تدريب نموذج جديد.
    أدوات التسجيل: أزرار لتسجيل الصوت أو تحميل ملف صوتي.
    أزرار الإجراءات: أزرار لتصنيف الصوت وحفظ النتائج.
    النتائج والسبيكتروغرام: يعرض نتائج التصنيف والسبيكتروغرام الخاص بالصوت.

الخطوة 1: اختيار اللغة

    استخدم القائمة المنسدلة في أعلى اليسار لاختيار اللغة المفضلة (العربية أو الإنجليزية).

الخطوة 2: اختيار طريقة التصنيف

    استخدام نموذج مدرب مسبقًا: اختر هذا الخيار لاستخدام نموذج Wav2Vec2 الافتراضي للتصنيف (موصى به لمعظم المستخدمين).
    تدريب نموذج جديد: اختر هذا الخيار إذا كنت ترغب في تدريب نموذج جديد. ستحتاج إلى:
        النقر على "تحميل بيانات التدريب" لاختيار ملف CSV يحتوي على مسارات الملفات الصوتية وتسمياتها (الصيغة: audio_path,label).
        النقر على "بدء التدريب" لتدريب النموذج (قد يستغرق ذلك بعض الوقت حسب جهازك).
        النقر على "حفظ النموذج المُدرب" لحفظ النموذج المدرب في مجلد.

الخطوة 3: تسجيل أو تحميل الصوت

    تسجيل الصوت:
        انقر على زر "تسجيل صوت" لبدء التسجيل.
        تحدث في الميكروفون.
        انقر على الزر مرة أخرى (الآن بعنوان "إيقاف التسجيل") للتوقف.
    تحميل الصوت:
        انقر على زر "تحميل ملف صوتي".
        اختر ملفًا بصيغة .wav أو .mp3 من جهازك.

الخطوة 4: تصنيف الصوت

    بعد التسجيل أو تحميل ملف صوتي، انقر على زر "تصنيف الصوت".
    سيقوم التطبيق بمعالجة الصوت وعرض:
        سبيكتروغرام الصوت على اليسار.
        نتائج التصنيف على اليمين، بما في ذلك:
            مدة الصوت.
            نسبة الإشارة إلى الضوضاء (SNR).
            المشاعر المتوقعة (سعيد، حزين، غاضب، محايد).
            درجة الثقة.
            توزيع الاحتمالات لجميع المشاعر.

الخطوة 5: حفظ النتائج

    بعد التصنيف، انقر على زر "حفظ النتائج".
    اختر موقعًا لحفظ النتائج كملف CSV.
    سيحتوي ملف CSV على مسار الصوت، الفئة المتوقعة، الثقة، المدة، وSNR.

الخطوة 6: تصفير التطبيق (اختياري)

    انقر على زر "تصفير" في الأعلى لمسح جميع البيانات وإعادة تحميل النموذج المدرب مسبقًا.

الخطوة 7: عرض معلومات "عن البرنامج"

    انقر على زر "عن البرنامج" لعرض معلومات حول التطبيق، بما في ذلك الإصدار وتفاصيل المطور.

6. استكشاف الأخطاء وإصلاحها

    فشل التطبيق في البدء:
        تأكد من تثبيت جميع المكتبات المطلوبة بشكل صحيح.
        تحقق مما إذا كان بايثون مضافًا إلى مسار النظام (PATH).
        تأكد من أن إصدار بايثون هو 3.8 أو أعلى.
    مشاكل في التسجيل:
        تأكد من أن الميكروفون متصل ويعمل.
        تحقق مما إذا كانت مكتبة sounddevice مثبتة بشكل صحيح.
    فشل تدريب النموذج:
        تأكد من أن ملف CSV لبيانات التدريب منسق بشكل صحيح.
        تحقق مما إذا كان لديك ذاكرة (RAM) كافية للتدريب.
    أداء بطيء:
        إذا كان التدريب أو التصنيف بطيئًا، فكر في استخدام وحدة معالجة رسوميات (GPU) مع دعم CUDA.
        قلل حجم الدفعة في دالة train_model إذا كان التدريب بطيئًا جدًا.

7. ملاحظات إضافية

    يقوم التطبيق تلقائيًا بحفظ الصوت المسجل باسم recorded_audio.wav في مجلد المشروع. يتم حذف هذا الملف عند تصفير التطبيق.
    تدريب نموذج جديد يتطلب مجموعة بيانات بصيغة CSV تحتوي على عمودين: audio_path (مسار الملف الصوتي) وlabel (واحد من: Happy، Sad، Angry، Neutral).
    يدعم التطبيق ملفات الصوت بصيغة .wav و.mp3 للتحميل.

خاتمة
تم تصميم تطبيق مصنف الصوت ليكون أداة سهلة الاستخدام لتصنيف المشاعر في الصوت. باتباع هذا الدليل، يمكنك تثبيت التطبيق، تشغيله، واستخدامه بسهولة. إذا واجهت أي مشاكل، راجع قسم استكشاف الأخطاء وإصلاحها أو تواصل مع المطور للحصول على الدعم.
الآن ،😊
