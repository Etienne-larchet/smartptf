from smartptf.utils.utils import force_list


def test_force_list():
    @force_list("a", "b")
    def test_func(a, b, c=None):
        return a, b, c

    # Test with single values
    assert test_func(1, 2) == ([1], [2], None)
    assert test_func(1, 2, c=3) == ([1], [2], 3)

    # Test with lists
    assert test_func([1], [2]) == ([1], [2], None)
    assert test_func([1], [2], c=[3]) == ([1], [2], [3])

    # Test with mixed types
    assert test_func(1, [2]) == ([1], [2], None)
    assert test_func(1, [2], c=3) == ([1], [2], 3)