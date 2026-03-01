import iris

def test_decorator_noop():
    # the decorator should be available and not alter the behavior
    @iris.offload(strategy="actor", return_type="int")
    def add(x, y):
        return x + y

    assert add(2, 3) == 5
    # ensure underlying Rust registration function exists
    assert hasattr(iris, "register_offload")


def test_offload_actor_execution():
    out = []

    @iris.offload(strategy="actor", return_type="int")
    def double(x):
        out.append(x * 2)
        return x * 2

    result = double(7)
    assert result == 14
    assert out == [14]

