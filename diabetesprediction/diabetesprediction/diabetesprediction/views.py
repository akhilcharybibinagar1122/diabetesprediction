import importlib.util
import logging
from pathlib import Path
import joblib
import pandas as pd
from django.conf import settings
from django.http import HttpResponseNotAllowed
from django.shortcuts import render

from .forms import DiabetesPredictionForm

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
]

DISCLAIMER_TEXT = (
    "Disclaimer: This tool is intended for educational purposes only and does not provide "
    "a clinical diagnosis or medical advice. The prediction does not confirm whether a person "
    "has diabetes. Please consult a qualified healthcare professional for any health concerns."
)

_CACHED_MODEL = None


def load_saved_model():
    model_path = Path(settings.BASE_DIR) / 'diabetes_model.joblib'
    if not model_path.exists():
        logger.warning("Model file not found at %s. Attempting to train model...", model_path)
        train_script = Path(settings.BASE_DIR) / 'train_model.py'
        if train_script.exists():
            spec = importlib.util.spec_from_file_location("train_model", train_script)
            tm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tm)
            tm.train_and_save_model(model_filename='diabetes_model.joblib')
        else:
            raise FileNotFoundError(
                f"Model file not found at {model_path} and training script not found at {train_script}"
            )

    loaded_model = joblib.load(model_path)
    steps = [step[0] for step in getattr(loaded_model, 'steps', [])]
    classes = getattr(loaded_model, 'classes_', None)
    logger.info("Loaded model from %s with steps: %s, classes: %s", model_path, steps, classes)
    return loaded_model


def get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
    _CACHED_MODEL = load_saved_model()
    return _CACHED_MODEL


def home(request):
    return render(request, 'home.html')


def predict(request):
    context = {
        'disclaimer': DISCLAIMER_TEXT,
    }
    return render(request, 'predict.html', context)


def result(request):
    if request.method != 'GET':
        return HttpResponseNotAllowed(['GET'])

    form_data = {
        'n1': request.GET.get('n1', ''),
        'n2': request.GET.get('n2', ''),
        'n3': request.GET.get('n3', ''),
        'n4': request.GET.get('n4', ''),
        'n5': request.GET.get('n5', ''),
        'n6': request.GET.get('n6', ''),
        'n7': request.GET.get('n7', ''),
        'n8': request.GET.get('n8', ''),
    }

    # If no parameters provided at all, prompt the user gracefully
    if not request.GET:
        return render(
            request,
            'predict.html',
            {
                'error_message': 'Please enter the required information to get a prediction.',
                'disclaimer': DISCLAIMER_TEXT,
                **form_data,
            },
            status=200,
        )

    form = DiabetesPredictionForm(request.GET)
    if not form.is_valid():
        first_error = None
        for field, errors in form.errors.items():
            if errors:
                first_error = errors[0]
                break
        error_message = first_error or "Invalid input. Please check the values provided."

        return render(
            request,
            'predict.html',
            {
                'form': form,
                'errors': form.errors,
                'error_message': error_message,
                'disclaimer': DISCLAIMER_TEXT,
                **form_data,
            },
            status=200,
        )

    # Input is valid: retrieve model and run inference
    try:
        model = get_model()
        if model is None:
            raise RuntimeError("Model instance could not be loaded.")

        features = form.get_features_in_order()
        input_data = pd.DataFrame([features], columns=FEATURE_NAMES)
        pred = model.predict(input_data)
        pred_class = int(pred[0])

        supported_classes = list(getattr(model, 'classes_', [0, 1]))
        if pred_class not in supported_classes:
            logger.warning(
                "Predicted class %s not in model supported classes %s",
                pred_class,
                supported_classes,
            )

        # Calculate probabilities only when supported by the loaded model
        probabilities = None
        if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
            try:
                raw_probs = model.predict_proba(input_data)[0]
                if hasattr(model, 'classes_'):
                    probabilities = {
                        cls: float(prob)
                        for cls, prob in zip(model.classes_, raw_probs)
                    }
                logger.debug(
                    "Inference completed: predicted_class=%d (%s), probabilities: %s",
                    pred_class,
                    "Positive" if pred_class == 1 else "Negative",
                    probabilities if probabilities is not None else raw_probs,
                )
            except Exception as pe:
                logger.warning(
                    "predict_proba could not be calculated: %s; using trusted predict() output.",
                    pe,
                )
        else:
            logger.debug(
                "Inference completed: predicted_class=%d (%s) without predict_proba support",
                pred_class,
                "Positive" if pred_class == 1 else "Negative",
            )

    except Exception as e:
        logger.exception("Prediction failed due to model error: %s", e)
        return render(
            request,
            'predict.html',
            {
                'error_message': 'The prediction service is currently unavailable. Please try again later.',
                'disclaimer': DISCLAIMER_TEXT,
                **form_data,
            },
            status=503,
        )

    result_label = "Positive" if pred_class == 1 else "Negative"

    return render(
        request,
        'predict.html',
        {
            'form': form,
            'result2': result_label,
            'disclaimer': DISCLAIMER_TEXT,
            **form_data,
        },
        status=200,
    )