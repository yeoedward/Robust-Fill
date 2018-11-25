from unittest import TestCase

from evaluate import evaluate
from sample_program import sample_program


class TestSample(TestCase):
    def test_sample_program_smoke_test(self):
        success = 0
        fail = 0
        for _ in range(1000):
            program = sample_program(10)
            try:
                transformed = evaluate(program, 'a b c d e 1 2 3 4 5')
                assert isinstance(transformed, str)
                success += 1
            except IndexError:
                # TODO: Remove after implementing string generation
                fail += 1
        print('Succeeded: {}, Failed: {}'.format(success, fail))
