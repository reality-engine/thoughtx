import json
import torch
import unittest
from generate_masks import generate_masks_from_embeddings
import os

# Root directory for test data
TEST_DATA_ROOT = "./handler/test_data/"


# (1) Load JSON files
def load_json_file(filename: str) -> torch.Tensor:
    """Load a tensor from a JSON file.

    Args:
    - filename (str): The path to the JSON file, relative to the test data root.

    Returns:
    - torch.Tensor: The tensor loaded from the file.
    """
    try:
        with open(TEST_DATA_ROOT + filename, 'r') as file:
            data = json.load(file)
        return torch.tensor(data)
    except FileNotFoundError:
        print("Current working directory:", os.getcwd())
        print("Failed to access:", TEST_DATA_ROOT + filename)
        raise

# (2) Define the test cases
test_cases = [
    {
        "description": "Test case 1",
        "input": load_json_file("input_embeddings_1.json"),
        "expected_mask": load_json_file("input_masks_1.json"),
        "expected_mask_invert": load_json_file("input_mask_invert_1.json")
    },
    {
        "description": "Not working example",
        "input": load_json_file("input_embeddings_1.json"),
        "expected_mask": load_json_file("input_masks_0.json"),
        "expected_mask_invert": load_json_file("input_mask_invert_0.json")
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
            
            # Check mask shape
            self.assertEqual(mask.shape, expected_mask.shape,
                             msg=f"Shapes differ for mask: {mask.shape} vs {expected_mask.shape}")

            # Check mask_invert shape
            self.assertEqual(mask_invert.shape, expected_mask_invert.shape,
                             msg=f"Shapes differ for mask invert: {mask_invert.shape} vs {expected_mask_invert.shape}")

            # Check mask content
            if not torch.equal(mask, expected_mask):
                diff_mask = (mask != expected_mask).nonzero(as_tuple=True)
                self.fail(f"Mask values differ at {diff_mask} for {test_case['description']}")

            # Check mask_invert content
            if not torch.equal(mask_invert, expected_mask_invert):
                diff_mask_invert = (mask_invert != expected_mask_invert).nonzero(as_tuple=True)
                self.fail(f"Mask invert values differ at {diff_mask_invert} for {test_case['description']}")


# (4) Run the tests
if __name__ == "__main__":
    unittest.main()
