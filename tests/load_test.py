"""
Load Testing for Mortgage AI API using Locust
Tests performance under concurrent user load
"""

from locust import HttpUser, task, between, events
import random
import json


class MortgageUser(HttpUser):
    """Simulated user making loan predictions."""

    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    weight = 10

    def on_start(self):
        """Login and get token."""
        self.client.headers = {"Content-Type": "application/json"}
        # In production, you'd login here and set auth token

    @task(5)
    def predict_loan(self):
        """Make loan prediction."""
        payload = {
            "income": random.randint(30000, 150000),
            "loan_amount": random.randint(5000, 100000),
            "interest_rate": round(random.uniform(2.0, 15.0), 2),
            "loan_term": random.randint(1, 30),
            "credit_score": random.randint(300, 850),
            "existing_loans": random.randint(0, 5),
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="predict"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success", False):
                    response.success()
                else:
                    response.failure("Prediction failed")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def batch_predict(self):
        """Make batch predictions."""
        applications = [
            {
                "income": random.randint(30000, 150000),
                "loan_amount": random.randint(5000, 100000),
                "interest_rate": round(random.uniform(2.0, 15.0), 2),
                "loan_term": random.randint(1, 30),
                "credit_score": random.randint(300, 850),
                "existing_loans": random.randint(0, 5),
            }
            for _ in range(random.randint(5, 20))
        ]

        with self.client.post(
            "/predict/batch",
            json={"applications": applications},
            catch_response=True,
            name="predict_batch"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    response.success()
                else:
                    response.failure("Batch prediction failed")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def get_explanation(self):
        """Get SHAP explanation."""
        payload = {
            "income": random.randint(30000, 150000),
            "loan_amount": random.randint(5000, 100000),
            "interest_rate": round(random.uniform(2.0, 15.0), 2),
            "loan_term": random.randint(1, 30),
            "credit_score": random.randint(300, 850),
            "existing_loans": random.randint(0, 5),
        }

        with self.client.post(
            "/explain",
            json=payload,
            catch_response=True,
            name="explain"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "shap_values" in data:
                    response.success()
                else:
                    response.failure("Explanation failed")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """Check API health."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="health"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Health check failed")
            else:
                response.failure(f"Status code: {response.status_code}")


class AdminUser(HttpUser):
    """Admin user performing heavy operations."""

    wait_time = between(5, 10)
    weight = 1

    @task(1)
    def get_metrics(self):
        """Get Prometheus metrics."""
        with self.client.get(
            "/metrics",
            catch_response=True,
            name="metrics"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class AuditorUser(HttpUser):
    """Auditor user checking logs."""

    wait_time = between(10, 30)
    weight = 2

    @task(1)
    def get_audit_logs(self):
        """Get audit logs."""
        with self.client.get(
            "/audit/logs?limit=100",
            catch_response=True,
            name="audit_logs"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


# Custom event handlers
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception):
    """Log slow requests."""
    if response_time > 5000:  # Log requests taking > 5 seconds
        print(f"SLOW REQUEST: {request_type} {name} took {response_time}ms")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary at end of test."""
    print("\n" + "=" * 50)
    print("LOAD TEST COMPLETE")
    print("=" * 50)

    stats = environment.stats
    print(f"\nTotal Requests: {stats.num_requests}")
    print(f"Failed Requests: {stats.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"RPS: {stats.total.total_rps:.2f}")


if __name__ == "__main__":
    import sys
    print("Run with: locust -f tests/load_test.py --host http://localhost:8000")
    print("Or: locust -f tests/load_test.py --host http://localhost:8000 --web-ui")
