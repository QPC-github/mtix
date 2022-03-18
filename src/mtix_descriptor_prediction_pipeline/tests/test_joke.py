from unittest import TestCase

import mtix_descriptor_prediction_pipeline

class TestJoke(TestCase):
    def test_is_string(self):
        s = mtix_descriptor_prediction_pipeline.joke()
        self.assertTrue(isinstance(s, str))