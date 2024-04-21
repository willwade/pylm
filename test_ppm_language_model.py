#call me like python -m unittest test_ppm_language_model.py

import unittest

from vocabulary import Vocabulary
from ppm_language_model import PPMLanguageModel, Context, Node


class TestPPMLanguageModel(unittest.TestCase):

    def setUp(self):
        # Setup for all tests
        self.vocab = Vocabulary()
        for symbol in ['a', 'b', 'c', 'd', 'e']:
            self.vocab.add_item(symbol)
        self.lm = PPMLanguageModel(self.vocab, max_order=5)
        self.lm.debug = True

    def test_predict_next_ids_valid_context(self):
        # Setup a valid context
        context = Context(self.lm.root, 1)
        self.lm.add_symbol_to_context(context, self.vocab.get_symbol_id('a'))
        self.lm.add_symbol_to_context(context, self.vocab.get_symbol_id('b'))
        
        # Predict next IDs
        predictions = self.lm.predict_next_ids(context, 3)
        self.assertTrue(predictions, "Should return predictions")
        self.assertEqual(len(predictions), 3, "Should return exactly three predictions")

        # Ensure predictions are in descending order of probabilities
        probs = [prob for _, prob in predictions]
        self.assertTrue(all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1)), "Probabilities should be in descending order")

    def test_predict_next_ids_empty_context(self):
        # Empty context
        context = Context(None, 0)
        predictions = self.lm.predict_next_ids(context, 3)
        self.assertEqual(predictions, [], "Should return an empty list for empty context")

    def test_predict_next_ids_invalid_context_order(self):
        # Invalid context order
        context = Context(self.lm.root, -1)
        predictions = self.lm.predict_next_ids(context, 3)
        self.assertEqual(predictions, [], "Should return an empty list for invalid context order")

    def test_predict_next_ids_no_predictions(self):
        # No predictions case
        context = Context(self.lm.root, 1)
        predictions = self.lm.predict_next_ids(context, 0)
        self.assertEqual(predictions, [], "Should return an empty list when number of predictions requested is zero")

        
    def test_add_symbol_to_node(self):
        vocab = Vocabulary()
        vocab.add_item('a')  # Assume add_symbol works correctly
        vocab.add_item('b')
        lm = PPMLanguageModel(vocab, max_order=5, debug=True)
        node = Node()
        
        # Test adding new symbol
        result_node = lm.add_symbol_to_node(node, vocab.get_id_or_oov('a'))
        self.assertIn(vocab.get_id_or_oov('a'), node.children)
        self.assertIsNotNone(result_node)
        self.assertEqual(result_node.count, 1)

        # Test incrementing existing symbol
        lm.add_symbol_to_node(node, vocab.get_id_or_oov('a'))
        self.assertEqual(node.children[vocab.get_id_or_oov('a')].count, 2)

    def test_find_appropriate_backoff(self):
        vocab = Vocabulary()
        vocab.add_item('a')
        lm = PPMLanguageModel(vocab, max_order=5)
        root = lm.root
        child = root.add_child(vocab.get_id_or_oov('a'))
        child_backoff = lm.find_appropriate_backoff(child, vocab.get_id_or_oov('a'))
        
        self.assertEqual(child_backoff, lm.root)  # Expect to fall back to root


    def test_get_probs(self):
        vocab = Vocabulary()
        vocab.add_item('a')
        lm = PPMLanguageModel(vocab, max_order=5, debug=True)
        root = lm.root
        a_id = vocab.get_id_or_oov('a')
        root.add_child(a_id)
        
        context = Context(root, 1)
        probs = lm.get_probs(context)
        
        # Verify probabilities add to 1
        self.assertAlmostEqual(sum(probs), 1.0)
        # Ensure the probability for 'a' is greater than 0
        self.assertGreater(probs[a_id], 0)


    def test_extreme_max_order(self):
        # Setup a large max_order that would typically exceed practical usage
        large_order_model = PPMLanguageModel(self.vocab, max_order=10000)
        context = large_order_model.create_context()
        test_input = "example input string"
        for symbol in test_input:
            large_order_model.add_symbol_and_update(context, ord(symbol))
        predictions = large_order_model.get_probs(context)
        self.assertNotEqual(len(predictions), 0, "Should handle high max_order without crashing")
        self.assertAlmostEqual(sum(predictions), 1.0, msg="Probabilities should sum to 1")

    def test_negative_max_order(self):
        """ Test model with a negative max_order value """
        with self.assertRaises(ValueError):
            PPMLanguageModel(self.vocab, -1)

    def test_empty_input(self):
        context = self.lm.create_context()  # Create an empty context
        predictions = self.lm.predict_next_ids(context, 5)
        self.assertEqual(len(predictions), 0, "Predictions should be empty for empty input")
    
    def test_handling_of_nonexistent_symbols(self):
        context = self.lm.create_context()
        nonexistent_symbol = self.vocab.size() + 100  # Assume this is outside the valid range
        self.lm.add_symbol_to_context(context, nonexistent_symbol)
        predictions = self.lm.predict_next_ids(context, 5)
        self.assertTrue(all(prob == 0 for _, prob in predictions), "All probabilities should be zero for a nonexistent symbol")
    
    def test_invalid_symbol(self):
        context = self.lm.create_context()
        invalid_symbol = -1  # Invalid symbol index
        self.lm.add_symbol_to_context(context, invalid_symbol)
        predictions = self.lm.predict_next_ids(context, 5)
        self.assertEqual(len(predictions), 0, "Predictions should be empty for invalid symbol")


    def test_predict_next_ids_valid_context(self):
        # Setup a valid context
        context = Context(self.lm.root, 1)
        self.lm.add_symbol_to_context(context, self.vocab.get_id_or_oov('a'))
        self.lm.add_symbol_to_context(context, self.vocab.get_id_or_oov('b'))
    
        # Predict next IDs
        predictions = self.lm.predict_next_ids(context, 3)
        self.assertTrue(predictions, "Should return predictions")
        self.assertEqual(len(predictions), 3, "Should return exactly three predictions")

        # Ensure predictions are in descending order of probabilities
        probs = [prob for _, prob in predictions]
        self.assertTrue(all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1)), "Probabilities should be in descending order")

    def test_predict_next_ids_empty_context(self):
        # Empty context
        context = Context(None, 0)
        predictions = self.lm.predict_next_ids(context, 3)
        self.assertEqual(predictions, [], "Should return an empty list for empty context")

    def test_predict_next_ids_invalid_context_order(self):
        # Invalid context order
        context = Context(self.lm.root, -1)
        predictions = self.lm.predict_next_ids(context, 3)
        self.assertEqual(predictions, [], "Should return an empty list for invalid context order")

    def test_predict_next_ids_no_predictions(self):
        # No predictions case
        context = Context(self.lm.root, 1)
        predictions = self.lm.predict_next_ids(context, 0)
        self.assertEqual(predictions, [], "Should return an empty list when number of predictions requested is zero")



