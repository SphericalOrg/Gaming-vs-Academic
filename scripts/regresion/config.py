RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3

TARGET = "grades"
EXCLUIR_MODELO = ["student_id", "age", "gender"]

COLUMNAS_ESPERADAS = [
    "student_id",
    "age",
    "gender",
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "gaming_genre",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
    "stress_level",
    "grades",
]

NUMERICAS_ESPERADAS = [
    "gaming_hours",
    "study_hours",
    "sleep_hours",
    "attendance",
    "social_activity",
    "device_usage",
    "reaction_time_ms",
    "addiction_score",
]

CATEGORICAS_ESPERADAS = ["gaming_genre", "stress_level"]
