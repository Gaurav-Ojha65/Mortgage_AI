"""
Capture Real UI Screenshots for Mortgage AI v3.1
================================================
Automates browser interactions using Selenium to capture 6 real, pixel-perfect
UI screenshots of the running application at 1440x900 viewport.

Output:
- docs/images/ui-dashboard.png
- docs/images/ui-predict-risk.png
- docs/images/ui-shap.png
- docs/images/ui-analytics.png
- docs/images/ui-what-if.png
- docs/images/ui-history.png
"""

import time
import json
import urllib.request
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

BASE_URL = "http://localhost:5173"
API_URL = "http://localhost:8001"
OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def seed_history():
    """Populate database with clean, realistic demonstration applications."""
    print("Seeding demonstration predictions via /analyze...")
    sample_applicants = [
        {"income": 85000, "loan_amount": 250000, "credit_score": 760, "loan_term": 30, "interest_rate": 6.0, "monthly_expenses": 2000, "dti_ratio": 0.28, "credit_utilization": 0.20},
        {"income": 58000, "loan_amount": 190000, "credit_score": 675, "loan_term": 30, "interest_rate": 6.8, "monthly_expenses": 2200, "dti_ratio": 0.38, "credit_utilization": 0.55},
        {"income": 45000, "loan_amount": 220000, "credit_score": 590, "loan_term": 30, "interest_rate": 8.5, "monthly_expenses": 2500, "dti_ratio": 0.52, "credit_utilization": 0.85},
        {"income": 110000, "loan_amount": 320000, "credit_score": 790, "loan_term": 15, "interest_rate": 5.5, "monthly_expenses": 2800, "dti_ratio": 0.25, "credit_utilization": 0.15},
        {"income": 62000, "loan_amount": 180000, "credit_score": 690, "loan_term": 30, "interest_rate": 6.5, "monthly_expenses": 2100, "dti_ratio": 0.34, "credit_utilization": 0.40},
    ]
    for app in sample_applicants:
        try:
            req = urllib.request.Request(
                f"{API_URL}/analyze",
                data=json.dumps(app).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                pass
        except Exception as e:
            print(f"Seed error for {app['credit_score']}: {e}")
    print("Seeding complete.")


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,900")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--force-device-scale-factor=1")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1440, 900)
    return driver


def authenticate(driver):
    driver.get(f"{BASE_URL}/login")
    time.sleep(1.5)
    driver.execute_script("""
        localStorage.setItem('mortgage_token', 'demo-session-token');
        localStorage.setItem('mortgage_user', JSON.stringify({username: 'admin', role: 'admin', name: 'Risk Underwriter'}));
    """)


def capture_all():
    seed_history()
    driver = get_driver()
    try:
        authenticate(driver)

        # -------------------------------------------------------------
        # 1. SCREENSHOT 1 — DASHBOARD (Hero)
        # -------------------------------------------------------------
        print("Capturing 1. Dashboard...")
        driver.get(f"{BASE_URL}/dashboard")
        time.sleep(3)
        driver.save_screenshot(str(OUTPUT_DIR / "ui-dashboard.png"))
        print("  [OK] ui-dashboard.png saved")

        # -------------------------------------------------------------
        # 2. SCREENSHOT 2 — PREDICT RISK (Borderline / Manual Review)
        # -------------------------------------------------------------
        print("Capturing 2. Predict Risk...")
        # Navigate via sidebar link to ensure fresh client-side state
        predict_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/predict')] | //span[text()='Predict Risk']")
        if predict_links:
            predict_links[0].click()
        else:
            driver.get(f"{BASE_URL}/predict")
        time.sleep(2)
        
        # Advance through multi-step form
        next_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'Next')]")
        if next_btns:
            next_btns[0].click()
            time.sleep(0.5)
            next_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'Next')]")
            if next_btns:
                next_btns[0].click()
                time.sleep(0.5)
        
        # Click Submit / Analyze Risk
        submit_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'Analyze Risk') or contains(text(), 'Analyze') or contains(text(), 'Submit')]")
        if submit_btns:
            submit_btns[0].click()
            # Wait for inference & SHAP calculation to finish rendering
            for _ in range(20):
                time.sleep(1)
                insights = driver.find_elements(By.XPATH, "//*[contains(text(), 'MODEL INSIGHTS') or contains(text(), 'Approval Probability') or contains(text(), 'RISK SCORE')]")
                if insights:
                    break
            time.sleep(2)
        
        driver.save_screenshot(str(OUTPUT_DIR / "ui-predict-risk.png"))
        print("  [OK] ui-predict-risk.png saved")

        # -------------------------------------------------------------
        # 3. SCREENSHOT 3 — SHAP (Explainability detail)
        # -------------------------------------------------------------
        print("Capturing 3. SHAP Explainability...")
        driver.execute_script("""
            var el = document.querySelector('.factors-section');
            if (el) el.scrollIntoView({behavior: 'instant', block: 'center'});
        """)
        time.sleep(1.5)
        driver.save_screenshot(str(OUTPUT_DIR / "ui-shap.png"))
        print("  [OK] ui-shap.png saved")

        # -------------------------------------------------------------
        # 4. SCREENSHOT 4 — ANALYTICS
        # -------------------------------------------------------------
        print("Capturing 4. Analytics...")
        driver.execute_script("""
            var page = document.querySelector('.page-container') || document.querySelector('.main-content') || window;
            if (page.scrollTo) page.scrollTo(0, 0);
        """)
        analytics_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/analytics')] | //span[text()='Analytics']")
        if analytics_links:
            analytics_links[0].click()
        else:
            driver.get(f"{BASE_URL}/analytics")
        time.sleep(3)
        driver.save_screenshot(str(OUTPUT_DIR / "ui-analytics.png"))
        print("  [OK] ui-analytics.png saved")

        # -------------------------------------------------------------
        # 5. SCREENSHOT 5 — WHAT-IF / SIMULATION
        # -------------------------------------------------------------
        print("Capturing 5. What-If Simulation...")
        predict_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/predict')] | //span[text()='Predict Risk']")
        if predict_links:
            predict_links[0].click()
        else:
            driver.get(f"{BASE_URL}/predict")
        time.sleep(2)
        next_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'Next')]")
        if next_btns:
            next_btns[0].click()
            time.sleep(0.5)
            next_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'Next')]")
            if next_btns:
                next_btns[0].click()
                time.sleep(0.5)
        submit_btns = driver.find_elements(By.XPATH, "//button[contains(text(), 'Analyze Risk') or contains(text(), 'Analyze') or contains(text(), 'Submit')]")
        if submit_btns:
            submit_btns[0].click()
            for _ in range(20):
                time.sleep(1)
                insights = driver.find_elements(By.XPATH, "//*[contains(text(), 'MODEL INSIGHTS') or contains(text(), 'Approval Probability')]")
                if insights:
                    break
            time.sleep(2)
        
        # Click "Simulate Changes" accordion
        sim_headers = driver.find_elements(By.XPATH, "//*[contains(text(), 'Simulate Changes')]")
        if sim_headers:
            sim_headers[0].click()
            time.sleep(1)
            # Set slider values via JS for positive scenario
            driver.execute_script("""
                var inputs = document.querySelectorAll('.simulator-panel input[type=range]');
                if (inputs.length >= 2) {
                    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                    nativeInputValueSetter.call(inputs[0], 790);
                    inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                    nativeInputValueSetter.call(inputs[1], 20);
                    inputs[1].dispatchEvent(new Event('input', { bubbles: true }));
                }
            """)
            time.sleep(0.5)
            sim_btns = driver.find_elements(By.XPATH, "//button[text()='Simulate']")
            if sim_btns:
                sim_btns[0].click()
                time.sleep(3)
        
        driver.execute_script("""
            var el = document.querySelector('.simulator-panel');
            if (el) el.scrollIntoView({behavior: 'instant', block: 'center'});
        """)
        time.sleep(1)
        driver.save_screenshot(str(OUTPUT_DIR / "ui-what-if.png"))
        print("  [OK] ui-what-if.png saved")

        # -------------------------------------------------------------
        # 6. SCREENSHOT 6 — HISTORY / AUDIT
        # -------------------------------------------------------------
        print("Capturing 6. History / Audit...")
        driver.execute_script("""
            var page = document.querySelector('.page-container') || document.querySelector('.main-content') || window;
            if (page.scrollTo) page.scrollTo(0, 0);
        """)
        time.sleep(0.5)
        history_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/history')] | //span[text()='History']")
        if history_links:
            history_links[0].click()
        else:
            driver.get(f"{BASE_URL}/history")
        time.sleep(3)
        driver.save_screenshot(str(OUTPUT_DIR / "ui-history.png"))
        print("  [OK] ui-history.png saved")

    finally:
        driver.quit()
        print("All 6 screenshots captured successfully!")


if __name__ == "__main__":
    capture_all()
