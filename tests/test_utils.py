from dltool.utils import flatten_dict, unflatten_dict


def test_flatten_dict():
    d = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": 4}
    assert flatten_dict(d) == {"a.b.c": 1, "a.b.d": 2, "a.e": 3, "f": 4}


def test_unflatten_dict():
    d = {"a.b.c": 1, "a.b.d": 2, "a.e": 3, "f": 4}
    assert unflatten_dict(d) == {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": 4}
    d = {"a": {"b.c": {"d.e": 1, "f": 2}, "g.h": {"j": 3}}}
    assert unflatten_dict(d) == {'a': {'b': {'c': {'d': {'e': 1}, 'f': 2}}, 'g': {'h': {'j': 3}}}}
    assert unflatten_dict(flatten_dict(d)) == unflatten_dict(d)