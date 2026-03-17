"""
Telegram Bot для подсчёта упражнений по видео
Использует MediaPipe Pose для анализа
"""

import os
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен бота (установи через переменную окружения)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")


class ExerciseAnalyzer:
    """Анализатор упражнений на видео"""

    EXERCISE_TYPES = {
        "pushups": {
            "name": "Отжимания 💪",
            "landmarks": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
            "down_angle": 90,
            "up_angle": 160,
        },
        "squats": {
            "name": "Приседания 🦵",
            "landmarks": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
            "down_angle": 90,
            "up_angle": 160,
        },
        "bicep_curls": {
            "name": "Подъём на бицепс 💪",
            "landmarks": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
            "down_angle": 160,
            "up_angle": 30,
            "inverted": True,
        },
    }

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_angle(self, a, b, c) -> float:
        """Вычисляет угол между тремя точками"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360.0 - angle if angle > 180.0 else angle

    def get_landmark_coords(self, landmarks, name, shape):
        """Получает координаты точки тела"""
        landmark = landmarks[self.mp_pose.PoseLandmark[name].value]
        return [landmark.x * shape[1], landmark.y * shape[0]]

    def analyze_video(self, video_path: str, exercise_type: str) -> dict:
        """Анализирует видео и считает упражнения"""
        if exercise_type not in self.EXERCISE_TYPES:
            return {"error": "Неизвестный тип упражнения"}

        config = self.EXERCISE_TYPES[exercise_type]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": "Не удалось открыть видео"}

        count = 0
        stage = "up"
        frames_processed = 0
        frames_with_pose = 0
        angles_history = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_processed += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                frames_with_pose += 1
                landmarks = results.pose_landmarks.landmark

                try:
                    points = [
                        self.get_landmark_coords(landmarks, config["landmarks"][0], frame.shape),
                        self.get_landmark_coords(landmarks, config["landmarks"][1], frame.shape),
                        self.get_landmark_coords(landmarks, config["landmarks"][2], frame.shape),
                    ]

                    angle = self.calculate_angle(*points)
                    angles_history.append(angle)

                    # Логика подсчёта
                    if config.get("inverted"):
                        # Для бицепса: down когда угол большой, up когда маленький
                        if angle > config["down_angle"]:
                            stage = "down"
                        if angle < config["up_angle"] and stage == "down":
                            stage = "up"
                            count += 1
                    else:
                        # Для отжиманий/приседаний
                        if angle > config["up_angle"]:
                            stage = "up"
                        if angle < config["down_angle"] and stage == "up":
                            stage = "down"
                            count += 1

                except (IndexError, KeyError):
                    continue

        cap.release()

        return {
            "exercise": config["name"],
            "count": count,
            "total_frames": total_frames,
            "frames_analyzed": frames_processed,
            "pose_detected_frames": frames_with_pose,
            "detection_rate": round(frames_with_pose / max(frames_processed, 1) * 100, 1),
            "duration_sec": round(total_frames / max(fps, 1), 1),
            "avg_angle": round(np.mean(angles_history), 1) if angles_history else 0,
        }


# Глобальный анализатор
analyzer = ExerciseAnalyzer()

# Хранилище выбора пользователя (в продакшене использовать Redis/DB)
user_exercise_choice = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🏋️ **Бот-счётчик упражнений**

Я могу посчитать количество упражнений на твоём видео!

**Как использовать:**
1. Отправь команду /analyze
2. Выбери тип упражнения
3. Отправь видео (до 20 МБ)
4. Получи результат!

**Поддерживаемые упражнения:**
• Отжимания 💪
• Приседания 🦵
• Подъём на бицепс 💪

**Советы для лучшего результата:**
📹 Снимай сбоку, чтобы было видно всё тело
💡 Хорошее освещение
🎯 Один человек в кадре
"""
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выбор типа упражнения"""
    keyboard = [
        [InlineKeyboardButton("💪 Отжимания", callback_data="exercise_pushups")],
        [InlineKeyboardButton("🦵 Приседания", callback_data="exercise_squats")],
        [InlineKeyboardButton("💪 Подъём на бицепс", callback_data="exercise_bicep_curls")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Выбери тип упражнения для анализа:",
        reply_markup=reply_markup
    )


async def exercise_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора упражнения"""
    query = update.callback_query
    await query.answer()

    exercise_type = query.data.replace("exercise_", "")
    user_id = query.from_user.id
    user_exercise_choice[user_id] = exercise_type

    exercise_name = ExerciseAnalyzer.EXERCISE_TYPES[exercise_type]["name"]
    await query.edit_message_text(
        f"✅ Выбрано: **{exercise_name}**\n\n"
        f"Теперь отправь видео для анализа 📹\n\n"
        f"_Видео должно быть до 20 МБ_",
        parse_mode="Markdown"
    )


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка видео"""
    user_id = update.message.from_user.id

    # Проверяем, выбрано ли упражнение
    if user_id not in user_exercise_choice:
        await update.message.reply_text(
            "⚠️ Сначала выбери тип упражнения!\n"
            "Используй команду /analyze"
        )
        return

    exercise_type = user_exercise_choice[user_id]
    exercise_name = ExerciseAnalyzer.EXERCISE_TYPES[exercise_type]["name"]

    # Отправляем сообщение о начале обработки
    processing_msg = await update.message.reply_text(
        f"⏳ Анализирую видео...\n"
        f"Упражнение: {exercise_name}\n\n"
        f"_Это может занять некоторое время_",
        parse_mode="Markdown"
    )

    try:
        # Скачиваем видео
        video = update.message.video or update.message.document
        file = await context.bot.get_file(video.file_id)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        # Анализируем
        result = analyzer.analyze_video(tmp_path, exercise_type)

        # Удаляем временный файл
        os.unlink(tmp_path)

        if "error" in result:
            await processing_msg.edit_text(f"❌ Ошибка: {result['error']}")
            return

        # Формируем ответ
        response = f"""
✅ **Анализ завершён!**

🏋️ **Упражнение:** {result['exercise']}
🔢 **Количество повторений:** {result['count']}

📊 **Статистика видео:**
• Длительность: {result['duration_sec']} сек
• Кадров обработано: {result['frames_analyzed']}
• Поза обнаружена: {result['detection_rate']}% кадров
• Средний угол: {result['avg_angle']}°

_Для нового анализа используй /analyze_
"""
        await processing_msg.edit_text(response, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        await processing_msg.edit_text(
            f"❌ Произошла ошибка при обработке видео.\n"
            f"Попробуй другое видео или формат."
        )

    # Очищаем выбор
    del user_exercise_choice[user_id]


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда помощи"""
    help_text = """
📖 **Справка по боту**

**Команды:**
/start - Приветствие и инструкция
/analyze - Начать анализ видео
/help - Эта справка

**Как снимать видео:**
1. Расположи камеру сбоку от себя
2. Убедись, что видно всё тело
3. Делай упражнения в умеренном темпе
4. Хорошее освещение улучшит результат

**Ограничения:**
• Максимальный размер видео: 20 МБ
• Форматы: MP4, MOV, AVI
• В кадре должен быть один человек
"""
    await update.message.reply_text(help_text, parse_mode="Markdown")


def main():
    """Запуск бота"""
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Установи TELEGRAM_BOT_TOKEN!")
        print("   export TELEGRAM_BOT_TOKEN='твой_токен'")
        return

    print("🚀 Запуск бота...")

    app = Application.builder().token(BOT_TOKEN).build()

    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(exercise_callback, pattern="^exercise_"))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))

    print("✅ Бот запущен! Нажми Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
