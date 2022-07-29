import collections
from unittest.mock import patch, _patch


MethodCall = collections.namedtuple(
    'MethodCall',
    (
        'args',
        'kwargs',
        'return_value',
    ),
)

class PatchInspector(object):
    def __init__(self, patch: _patch, orig: callable):
        self.patch = patch
        self.orig = orig

    def __enter__(self):
        self.calls = []

        def wrapper(*a, **kw):

            result = self.orig(*a, **kw)
            self.calls.append(
                MethodCall(
                    args=a,
                    kwargs=kw,
                    return_value=result,
                )
            )
            return result

        self.patch.start().side_effect = wrapper
        return self
    
    def __exit__(self, *a):
        self.patch.stop()

    def assertCalledTimes(self, n):
        if len(self.calls) != n:
            raise AssertionError(f"Number of calls to {self.patch} was {len(self.calls)} != {n}")


class MethodInspector(object):
    """Wrap a method and track calls to allow making assertions about them.
    """

    def __init__(self, klass, method_name):

        self.klass = klass
        self.method_name = method_name
        self.orig = getattr(klass, method_name)
        self.inspector = PatchInspector(
            patch.object(self.klass, self.method_name, autospec=True),
            self.orig
        )

    def __enter__(self):
        self.inspector.patch = patch.object(self.klass, self.method_name, autospec=True)    
        return self.inspector.__enter__()

    def __exit__(self, *a):
        self.inspector.__exit__(*a)