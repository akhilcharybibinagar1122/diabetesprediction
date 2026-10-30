import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'diabetesprediction.settings')
django.setup()

from django.test import Client
from django.test.utils import setup_test_environment

setup_test_environment()

c = Client()

print("--- 1. Testing Home Page (/) ---")
r_home = c.get('/')
print(f"Status: {r_home.status_code}")
assert r_home.status_code == 200, "Home page failed"

print("\n--- 2. Testing Predict Page (/predict/) ---")
r_predict = c.get('/predict/')
print(f"Status: {r_predict.status_code}")
assert r_predict.status_code == 200, "Predict page failed"
assert 'name="n1"' in r_predict.content.decode('utf-8')
assert 'name="n1>"' not in r_predict.content.decode('utf-8')

print("\n--- 3. Testing Sample 1 (Non-diabetic profile: outcome 0) ---")
# Pregnancies=1, Glucose=85, BP=66, Skin=29, Insulin=0, BMI=26.6, Pedigree=0.351, Age=31
r_neg = c.get('/predict/result?n1=1&n2=85&n3=66&n4=29&n5=0&n6=26.6&n7=0.351&n8=31')
print(f"Status: {r_neg.status_code}, result2: {r_neg.context.get('result2')}")
assert r_neg.status_code == 200
assert r_neg.context.get('result2') == 'Negative'

print("\n--- 4. Testing Sample 2 (Diabetic profile: outcome 1) ---")
# Pregnancies=6, Glucose=148, BP=72, Skin=35, Insulin=0, BMI=33.6, Pedigree=0.627, Age=50
r_pos = c.get('/predict/result?n1=6&n2=148&n3=72&n4=35&n5=0&n6=33.6&n7=0.627&n8=50')
print(f"Status: {r_pos.status_code}, result2: {r_pos.context.get('result2')}")
assert r_pos.status_code == 200
assert r_pos.context.get('result2') == 'Positive'

print("\n--- 5. Testing Prompt Query Sample ---")
# n1=12&n2=11&n3=1&n4=55&n5=44&n6=11&n7=23&n8=33
r_prompt = c.get('/predict/result?n1=12&n2=11&n3=1&n4=55&n5=44&n6=11&n7=23&n8=33')
print(f"Status: {r_prompt.status_code}, result2: {r_prompt.context.get('result2')}")
assert r_prompt.status_code == 200
assert r_prompt.context.get('result2') in ['Positive', 'Negative']

print("\n--- 6. Testing Invalid Query (Missing n1, non-numeric, nan) ---")
r_missing = c.get('/predict/result?n2=100&n3=70&n4=20&n5=80&n6=25&n7=0.5&n8=30')
print(f"Missing n1 status: {r_missing.status_code}, error: {r_missing.context.get('error_message')}")
assert r_missing.status_code == 200
assert "Pregnancies is required." in r_missing.context.get('error_message')

r_nan = c.get('/predict/result?n1=1&n2=nan&n3=70&n4=20&n5=80&n6=25&n7=0.5&n8=30')
print(f"NaN status: {r_nan.status_code}, error: {r_nan.context.get('error_message')}")
assert r_nan.status_code == 200
assert "Glucose must be a valid number." in r_nan.context.get('error_message')

print("\n--- 7. Checking Medical Disclaimer on Result Page ---")
assert "Disclaimer:" in r_pos.content.decode('utf-8')
assert "educational purposes only" in r_pos.content.decode('utf-8')

print("\nAll workflow and integration checks PASSED successfully!")
