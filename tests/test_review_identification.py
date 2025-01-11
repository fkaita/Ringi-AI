import unittest

from llm import create_client
from review import perform_identification

class TestPerformIdentificationRealAPI(unittest.TestCase):
    def test_perform_identification(self):
        import os
        from dotenv import load_dotenv
        load_dotenv()  # Load variables from .env
        api_key = os.getenv("OPENAI_API_KEY")

        # Create a real client
        client, model = create_client("gpt-4o-2024-05-13")

        # Inputs
        text = "I want to install Sansan business card management system."
        num_reflections = 1
        num_fs_examples = 1
        num_reviews_ensemble = 1
        temperature = 0.75
        msg_history = None


        # Call the function
        result = perform_identification(
            text=text,
            model=model,
            client=client,
            num_reflections=num_reflections,
            num_fs_examples=num_fs_examples,
            num_reviews_ensemble=num_reviews_ensemble,
            temperature=temperature,
            msg_history=msg_history
        )

        # Print the result
        print("Review Result:", result)

        # Assertions
        # self.assertIsInstance(result, dict)  # Ensure the output is a dictionary
        # self.assertIn("mock_key", result)  # Update based on expected key in result

if __name__ == "__main__":
    unittest.main()
