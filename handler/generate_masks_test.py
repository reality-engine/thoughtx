import json
import torch
import unittest
from handler.generate_masks import generate_masks_from_embeddings

# (1) Load JSON files
def load_json_file(filename: str) -> torch.Tensor:
    """Load a tensor from a JSON file.

    Args:
    - filename (str): The path to the JSON file.

    Returns:
    - torch.Tensor: The tensor loaded from the file.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return torch.tensor(data)

# (2) Define the test cases
test_cases = [
    {
        "description": "Test case 1",
        "input": load_json_file("/test_data/input_embeddings_1.json"),
        "expected_mask": load_json_file("/test_data/input_masks_1.json"),
        "expected_mask_invert": load_json_file("/test_data/input_mask_invert_1.json")
    },
    # The following test case is a placeholder. Please ensure "/mnt/data/input_embeddings_2.json" and the other relevant files exist.
    {
        "description": "Not working example",
        "input": load_json_file("/test_data/input_embeddings_1.json"),
        "expected_mask": load_json_file("/test_data/input_masks_0.json"),
        "expected_mask_invert": load_json_file("/test_data/input_mask_invert_0.json")
    }
]

# (3) Define the unit test
class TestGenerateMasksFromEmbeddings(unittest.TestCase):
    """Unit test for the function generate_masks_from_embeddings."""
    
    def test_masks_generation(self):
        """Test the generation of attention masks from embeddings."""
        for test_case in test_cases:
            input_embeddings = test_case["input"]
            expected_mask = test_case["expected_mask"]
            expected_mask_invert = test_case["expected_mask_invert"]

            # Call the function
            mask, mask_invert = generate_masks_from_embeddings(input_embeddings)
            
            # Assert that the masks match the expected values
            self.assertTrue(torch.equal(mask, expected_mask))
            self.assertTrue(torch.equal(mask_invert, expected_mask_invert))

# (4) Run the tests
if __name__ == "__main__":
    unittest.main()
