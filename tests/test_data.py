from cvlab.data import Product, pattern_from_name, split_by_id


def test_pattern_keywords_require_exactly_one_match():
    assert pattern_from_name("Roadster Men Striped Blue Shirt") == "striped"
    assert pattern_from_name("Turtle Check Men Navy Blue Shirt") == "checked"
    assert pattern_from_name("Puma Men Printed Black T-shirt") == "printed"
    assert pattern_from_name("Manchester United Men Solid Black Track Pants") == "solid"
    assert pattern_from_name("Puma Men Grey T-shirt") is None
    assert pattern_from_name("Striped and Printed Top") is None  # ambiguous -> unlabeled
    assert pattern_from_name("Checkered Shorts") is None  # whole-word match only


def _p(i):
    return Product(i, f"p{i}", "Tops", "Black", "Women", "Casual", None, b"")


def test_split_by_id_is_disjoint_and_deterministic():
    products = [_p(i) for i in range(100)]
    tr, te = split_by_id(products)
    assert not {products[i].id for i in tr} & {products[i].id for i in te}
    assert len(te) == 30
    assert split_by_id(products) == (tr, te)


def test_attributes_include_pattern_only_when_known():
    p = _p(1)
    assert "pattern" not in p.attributes()
    p.pattern = "striped"
    assert p.attributes()["pattern"] == "striped"
