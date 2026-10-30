import math
from django import forms


class DiabetesPredictionForm(forms.Form):
    n1 = forms.FloatField(
        label="Pregnancies",
        min_value=0,
        max_value=30,
        required=True,
        error_messages={
            'required': 'Pregnancies is required.',
            'invalid': 'Pregnancies must be a valid number.',
            'min_value': 'Pregnancies cannot be negative.',
            'max_value': 'Pregnancies must be 30 or less.',
        }
    )
    n2 = forms.FloatField(
        label="Glucose",
        min_value=0,
        max_value=500,
        required=True,
        error_messages={
            'required': 'Glucose is required.',
            'invalid': 'Glucose must be a valid number.',
            'min_value': 'Glucose cannot be negative.',
            'max_value': 'Glucose must be 500 or less.',
        }
    )
    n3 = forms.FloatField(
        label="Blood Pressure",
        min_value=0,
        max_value=300,
        required=True,
        error_messages={
            'required': 'Blood Pressure is required.',
            'invalid': 'Blood Pressure must be a valid number.',
            'min_value': 'Blood Pressure cannot be negative.',
            'max_value': 'Blood Pressure must be 300 or less.',
        }
    )
    n4 = forms.FloatField(
        label="Skin Thickness",
        min_value=0,
        max_value=150,
        required=True,
        error_messages={
            'required': 'Skin Thickness is required.',
            'invalid': 'Skin Thickness must be a valid number.',
            'min_value': 'Skin Thickness cannot be negative.',
            'max_value': 'Skin Thickness must be 150 or less.',
        }
    )
    n5 = forms.FloatField(
        label="Insulin",
        min_value=0,
        max_value=1000,
        required=True,
        error_messages={
            'required': 'Insulin is required.',
            'invalid': 'Insulin must be a valid number.',
            'min_value': 'Insulin cannot be negative.',
            'max_value': 'Insulin must be 1000 or less.',
        }
    )
    n6 = forms.FloatField(
        label="BMI",
        min_value=0,
        max_value=100,
        required=True,
        error_messages={
            'required': 'BMI is required.',
            'invalid': 'BMI must be a valid number.',
            'min_value': 'BMI cannot be negative.',
            'max_value': 'BMI must be 100 or less.',
        }
    )
    n7 = forms.FloatField(
        label="Diabetes Pedigree Function",
        min_value=0,
        max_value=100,
        required=True,
        error_messages={
            'required': 'Diabetes Pedigree function is required.',
            'invalid': 'Diabetes Pedigree function must be a valid number.',
            'min_value': 'Diabetes Pedigree function cannot be negative.',
            'max_value': 'Diabetes Pedigree function must be 100 or less.',
        }
    )
    n8 = forms.FloatField(
        label="Age",
        min_value=1,
        max_value=120,
        required=True,
        error_messages={
            'required': 'Age is required.',
            'invalid': 'Age must be a valid number.',
            'min_value': 'Age must be at least 1.',
            'max_value': 'Age must be 120 or less.',
        }
    )

    def clean(self):
        cleaned_data = super().clean()
        for field_name in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']:
            val = cleaned_data.get(field_name)
            if val is not None:
                if not isinstance(val, (int, float)) or not math.isfinite(val):
                    self.add_error(field_name, f"{self.fields[field_name].label} must be a valid finite number.")
        return cleaned_data

    def get_features_in_order(self):
        """Returns the 8 numeric features in the exact order expected by the model."""
        return [
            float(self.cleaned_data['n1']),
            float(self.cleaned_data['n2']),
            float(self.cleaned_data['n3']),
            float(self.cleaned_data['n4']),
            float(self.cleaned_data['n5']),
            float(self.cleaned_data['n6']),
            float(self.cleaned_data['n7']),
            float(self.cleaned_data['n8']),
        ]
