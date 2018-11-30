from unittest import TestCase

from sample import sample_example


class TestSample(TestCase):
    def test_sample_example_smoke_test(self):
        num_discarded_programs = 0
        num_tries = 100
        for _ in range(num_tries):
            example = sample_example()
            num_discarded_programs += example.num_discarded_programs

        # Seems to be at between 10-20%
        print('Number of discarded programs: {}/{}'.format(
            num_discarded_programs,
            num_tries,
        ))
