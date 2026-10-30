import math
from pathlib import Path
from unittest.mock import MagicMock, patch
from django.conf import settings
from django.test import SimpleTestCase
import joblib
import pandas as pd

from diabetesprediction.forms import DiabetesPredictionForm
from diabetesprediction import views
import train_model


class DiabetesPredictionComprehensiveTests(SimpleTestCase):
    def setUp(self):
        # Contrasting input profiles
        self.negative_sample = {
            'n1': '1',     # Pregnancies
            'n2': '85',    # Glucose (low/normal)
            'n3': '66',    # BloodPressure
            'n4': '29',    # SkinThickness
            'n5': '0',     # Insulin
            'n6': '26.6',  # BMI
            'n7': '0.351', # DiabetesPedigreeFunction
            'n8': '31',    # Age
        }

        self.positive_sample = {
            'n1': '6',     # Pregnancies
            'n2': '148',   # Glucose (elevated)
            'n3': '72',    # BloodPressure
            'n4': '35',    # SkinThickness
            'n5': '0',     # Insulin
            'n6': '33.6',  # BMI (obese)
            'n7': '0.627', # DiabetesPedigreeFunction
            'n8': '50',    # Age
        }

    # 1. The saved model loads from the intended path.
    def test_saved_model_loads_from_intended_path(self):
        intended_path = Path(settings.BASE_DIR) / 'diabetes_model.joblib'
        self.assertTrue(intended_path.exists(), f"Model file must exist at {intended_path}")
        loaded = joblib.load(intended_path)
        self.assertIsNotNone(loaded)
        self.assertTrue(hasattr(loaded, 'predict'), "Loaded object must have a predict method")

    # 2. The loaded estimator has the expected features.
    def test_loaded_estimator_has_expected_features(self):
        intended_path = Path(settings.BASE_DIR) / 'diabetes_model.joblib'
        model = joblib.load(intended_path)
        scaler = model.named_steps.get('scaler')
        classifier = model.named_steps.get('classifier')

        self.assertIsNotNone(scaler, "Pipeline must contain 'scaler'")
        self.assertIsNotNone(classifier, "Pipeline must contain 'classifier'")
        self.assertEqual(scaler.n_features_in_, 8, "Scaler must expect 8 features")
        self.assertEqual(classifier.n_features_in_, 8, "Classifier must expect 8 features")
        self.assertEqual(len(views.FEATURE_NAMES), 8, "View FEATURE_NAMES must contain 8 features")

    # 3. The expected class labels are present.
    def test_expected_class_labels_are_present(self):
        intended_path = Path(settings.BASE_DIR) / 'diabetes_model.joblib'
        model = joblib.load(intended_path)
        self.assertTrue(hasattr(model, 'classes_'), "Model must have classes_ attribute")
        self.assertEqual(list(model.classes_), [0, 1], "Model classes_ must be [0, 1]")

    # 4. The prediction view uses the actual estimator output (contrasting inputs).
    def test_prediction_view_uses_actual_estimator_output(self):
        # Non-diabetic profile produces Negative
        resp_neg = self.client.get('/predict/result', self.negative_sample)
        self.assertEqual(resp_neg.status_code, 200)
        self.assertEqual(resp_neg.context['result2'], 'Negative')
        self.assertContains(resp_neg, 'Result: Negative')

        # Diabetic profile produces Positive
        resp_pos = self.client.get('/predict/result', self.positive_sample)
        self.assertEqual(resp_pos.status_code, 200)
        self.assertEqual(resp_pos.context['result2'], 'Positive')
        self.assertContains(resp_pos, 'Result: Positive')

    # 5. A model output of 0 renders the negative label.
    @patch('diabetesprediction.views.get_model')
    def test_model_output_0_renders_negative_label(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.classes_ = [0, 1]
        mock_get_model.return_value = mock_model

        response = self.client.get('/predict/result', self.positive_sample)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['result2'], 'Negative')
        self.assertContains(response, 'Result: Negative')
        self.assertNotContains(response, 'Result: Positive')

    # 6. A model output of 1 renders the positive label.
    @patch('diabetesprediction.views.get_model')
    def test_model_output_1_renders_positive_label(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.classes_ = [0, 1]
        mock_get_model.return_value = mock_model

        response = self.client.get('/predict/result', self.negative_sample)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['result2'], 'Positive')
        self.assertContains(response, 'Result: Positive')
        self.assertNotContains(response, 'Result: Negative')

    # 7. Missing and invalid form inputs are rejected safely.
    def test_missing_and_invalid_form_inputs_rejected_safely(self):
        # Missing required field n1
        missing_n1 = self.negative_sample.copy()
        del missing_n1['n1']
        resp = self.client.get('/predict/result', missing_n1)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('error_message', resp.context)
        self.assertContains(resp, 'Pregnancies is required.')

        # Empty field
        empty_field = self.negative_sample.copy()
        empty_field['n2'] = ''
        resp = self.client.get('/predict/result', empty_field)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('error_message', resp.context)

        # Non-numeric field
        non_num = self.negative_sample.copy()
        non_num['n3'] = 'abc'
        resp = self.client.get('/predict/result', non_num)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('error_message', resp.context)

        # NaN input
        nan_param = self.negative_sample.copy()
        nan_param['n4'] = 'nan'
        resp = self.client.get('/predict/result', nan_param)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('error_message', resp.context)

        # Infinite input
        inf_param = self.negative_sample.copy()
        inf_param['n5'] = 'inf'
        resp = self.client.get('/predict/result', inf_param)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('error_message', resp.context)

        # Negative input
        neg_val = self.negative_sample.copy()
        neg_val['n1'] = '-1'
        resp = self.client.get('/predict/result', neg_val)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('error_message', resp.context)

    # 8. The result page displays the prediction generated by the model.
    def test_result_page_displays_prediction_generated_by_model(self):
        resp = self.client.get('/predict/result', self.negative_sample)
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, 'predict.html')
        self.assertContains(resp, 'Result: Negative')
        self.assertContains(resp, 'Disclaimer:')
        self.assertContains(resp, 'educational purposes only')

    # 9. A model-loading failure is handled appropriately.
    @patch('diabetesprediction.views.get_model')
    def test_model_loading_failure_handled_appropriately(self, mock_get_model):
        mock_get_model.side_effect = RuntimeError("Simulated missing or corrupted model")
        resp = self.client.get('/predict/result', self.negative_sample)
        self.assertEqual(resp.status_code, 503)
        self.assertNotEqual(resp.status_code, 500)
        self.assertContains(resp, 'currently unavailable', status_code=503)
        self.assertNotContains(resp, 'Simulated missing or corrupted model', status_code=503)
        self.assertNotContains(resp, 'Traceback', status_code=503)

    # 10. The training and inference feature orders agree.
    def test_training_and_inference_feature_orders_agree(self):
        expected_order = [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age',
        ]
        self.assertEqual(views.FEATURE_NAMES, expected_order)
        self.assertEqual(train_model.FEATURE_NAMES, expected_order)

        # Form extraction order
        form = DiabetesPredictionForm({
            'n1': 1, 'n2': 2, 'n3': 3, 'n4': 4,
            'n5': 5, 'n6': 6, 'n7': 7, 'n8': 8
        })
        self.assertTrue(form.is_valid())
        self.assertEqual(form.get_features_in_order(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    # 11. Home page and predict form render successfully
    def test_routes_and_navigation(self):
        resp_home = self.client.get('/')
        self.assertEqual(resp_home.status_code, 200)
        self.assertTemplateUsed(resp_home, 'home.html')

        resp_predict = self.client.get('/predict/')
        self.assertEqual(resp_predict.status_code, 200)
        self.assertTemplateUsed(resp_predict, 'predict.html')

        # Form field names
        html = resp_predict.content.decode('utf-8')
        for i in range(1, 9):
            self.assertIn(f'name="n{i}"', html)
            self.assertNotIn(f'name="n{i}>"', html)

        # Parameterless result access handled gracefully
        resp_paramless = self.client.get('/predict/result')
        self.assertEqual(resp_paramless.status_code, 200)

        # POST rejected
        resp_post = self.client.post('/predict/result', self.negative_sample)
        self.assertEqual(resp_post.status_code, 405)

    # 12. Probability values correspond to correct class labels
    def test_predict_proba_corresponds_to_correct_class_labels(self):
        intended_path = Path(settings.BASE_DIR) / 'diabetes_model.joblib'
        model = joblib.load(intended_path)
        self.assertTrue(hasattr(model, 'predict_proba'), "Model must support predict_proba")

        sample_df = pd.DataFrame(
            [[1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0]],
            columns=views.FEATURE_NAMES,
        )
        probs = model.predict_proba(sample_df)[0]
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)
        # Class 0 should have higher probability for healthy profile
        self.assertGreater(probs[0], probs[1])

    # 13. Resilience if predict_proba raises an attribute error
    @patch('diabetesprediction.views.get_model')
    def test_predict_proba_failure_does_not_break_prediction(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.classes_ = [0, 1]
        mock_model.predict_proba.side_effect = AttributeError(
            "'LogisticRegression' object has no attribute 'multi_class'"
        )
        mock_get_model.return_value = mock_model

        response = self.client.get('/predict/result', self.negative_sample)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['result2'], 'Negative')
        self.assertContains(response, 'Result: Negative')
